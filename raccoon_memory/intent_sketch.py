"""Read-only experimental intent sketches and context binding; never dispatch.

The model sees only the transcript. This module does not infer actions or fill
quantities from the transcript. Context and opaque IDs are caller-owned.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema

from . import task_program as v1
from . import task_program_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / 'configs/intent_sketch.v1.schema.json'
PROMPT_PATH = ROOT / 'configs/intent_sketch.v1.prompt.txt'

TASKS = {'move', 'stack', 'unstack', 'swap', 'remember', 'register', 'arrange'}
TRANSFER = {'move', 'stack', 'unstack'}
REFERENCE_KINDS = {
    'return': {'arm_home', 'registered_initial', 'task_start', 'learned_memory', 'deictic', 'unresolved'},
    'repeat': {'active_task', 'previous_task', 'previous_program', 'prior_steps', 'deictic', 'unresolved'},
    'continue': {'active_task', 'previous_task', 'deictic', 'unresolved'},
}
ARGUMENTS = {
    'move': {'source_slot', 'destination_slot', 'object_id', 'count', 'height'},
    'stack': {'source_slot', 'destination_slot', 'object_id', 'count', 'height'},
    'unstack': {'source_slot', 'destination_slot', 'object_id', 'count', 'height'},
    'swap': {'source_slot', 'destination_slot'},
    'remember': set(), 'register': {'layout'}, 'arrange': {'layout'},
    'repeat': {'repeat_mode', 'repeat_count'}, 'stop_motion': set(),
    'stop_repeat': {'boundary'}, 'continue': set(), 'return': set(),
    'status': {'subject'}, 'clarify': {'reason', 'question'}, 'question': set(),
}


class SketchError(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail
        super().__init__(f'{code}: {detail}')


def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))


def prompt():
    return PROMPT_PATH.read_text(encoding='utf-8')


def build_messages(transcript):
    if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > v1.MAX_TEXT_CHARS:
        raise SketchError('invalid_input', 'expected a nonempty transcript within the existing length limit')
    return [{'role':'system', 'content':prompt()},
            {'role':'user', 'content':json.dumps({'transcript':transcript}, ensure_ascii=False, separators=(',',':'))}]


def _same_value(left,right):
    return type(left) is type(right) and left==right


def validate_sketch(sketch, transcript):
    if not isinstance(transcript, str) or not transcript.strip():
        raise SketchError('invalid_input', 'transcript is empty')
    validator = jsonschema.Draft202012Validator(schema())
    errors = sorted(validator.iter_errors(sketch), key=lambda e: list(map(str,e.absolute_path)))
    if errors:
        raise SketchError('invalid_schema', errors[0].message)
    requested_count=sum(step['polarity']=='request' for step in sketch['steps'])
    auxiliary_count=len(sketch['steps'])-requested_count
    if requested_count>8 or auxiliary_count>8:
        raise SketchError('semantic_capacity_exceeded',
                          f'{requested_count} requested and {auxiliary_count} auxiliary records exceed separate limits of 8')
    for index, step in enumerate(sketch['steps']):
        op, polarity, args = step['op'], step['polarity'], step.get('args', {})
        if step['evidence'] not in transcript:
            raise SketchError('ungrounded_evidence', f'step {index} evidence is absent from the transcript')
        extra = set(args) - ARGUMENTS[op]
        if extra:
            raise SketchError('invalid_arguments', f'step {index} has unsupported fields: {sorted(extra)}')
        reference = step.get('reference')
        if reference is not None:
            if op not in REFERENCE_KINDS or reference['kind'] not in REFERENCE_KINDS[op]:
                raise SketchError('invalid_reference', f'step {index} reference kind is not valid for {op}')
            if reference['mention'] not in transcript:
                raise SketchError('ungrounded_reference', f'step {index} reference is absent from the transcript')
            if reference['kind'] == 'prior_steps':
                indices = reference.get('step_indices')
                if not indices or indices != sorted(set(indices)) or any(i >= index for i in indices):
                    raise SketchError('invalid_reference', f'step {index} must point to distinct earlier steps')
                if any(sketch['steps'][i]['polarity'] != 'request' or
                       sketch['steps'][i]['op'] not in TASKS | {'return'} for i in indices):
                    raise SketchError('invalid_reference', f'step {index} refers to a non-task or excluded step')
            elif 'step_indices' in reference:
                raise SketchError('invalid_reference', f'step {index} has indices without prior_steps')
        if polarity == 'request' and op in REFERENCE_KINDS and reference is None:
            raise SketchError('missing_reference', f'step {index} did not express its reference')
        if op == 'question' and polarity != 'query':
            raise SketchError('invalid_polarity', 'a question is not an action')
        if op == 'clarify' and polarity != 'uncertain':
            raise SketchError('invalid_polarity', 'clarification is not an action')
        if polarity == 'request' and op == 'repeat':
            mode = args.get('repeat_mode')
            if mode is None:
                raise SketchError('missing_repeat_mode', f'step {index} has no repeat mode')
            if mode in ('continuous','unresolved') and 'repeat_count' in args:
                raise SketchError('invalid_repeat_count', f'step {index} has a count for {mode}')
            if mode in ('total','additional') and 'repeat_count' not in args:
                raise SketchError('missing_repeat_count', f'step {index} has no stated count')
        for ex in step.get('exclusions', []):
            if ex['evidence'] not in transcript:
                raise SketchError('ungrounded_exclusion', f'step {index} exclusion is absent from the transcript')
            if (ex['field']=='reference') != isinstance(ex['value'],str):
                raise SketchError('invalid_exclusion', f'step {index} exclusion has the wrong value type')
            if ex['field'] in args and _same_value(args[ex['field']],ex['value']):
                raise SketchError('contradictory_correction', f'step {index} both excludes and chooses the same value')
            if ex['field'] == 'reference' and reference is not None and reference['kind'] == ex['value']:
                raise SketchError('contradictory_correction', f'step {index} both excludes and chooses the same reference')
    return copy.deepcopy(sketch)


def _bind_reference(op, ref, context):
    """Return an interpretation and provenance, not a physical action."""
    kind = ref['kind']
    detail = {'requested_kind':kind, 'mention':ref['mention'], 'bound_kind':kind,
              'status':'literal', 'required_checks':[]}
    if kind == 'deictic' and op == 'return':
        focus = context.get('return_focus')
        if focus is None:
            detail.update(bound_kind='unresolved',status='ambiguous_or_missing')
        else:
            detail.update(bound_kind=focus['kind'],status='bound_to_recorded_focus',
                          focus_revision=focus['revision'],focus_source=focus['source'])
            detail['required_checks'].append('return_focus_and_state_still_current')
    elif kind == 'deictic' and op in ('repeat','continue'):
        # A missing prior-reference record does not prove there is no prior task.
        prior = context.get('previous_task')
        running = context['execution_state'] == 'running'
        idle = context['execution_state'] in ('idle','stopped_confirmed')
        if running and prior is not None and prior['status'] == 'none':
            kind = 'active_task'
        elif idle and prior is not None and prior['status'] == 'unique':
            kind = 'previous_task'
        else:
            detail.update(bound_kind='unresolved',status='ambiguous_or_missing')
            return detail
        detail.update(bound_kind=kind,status='bound_to_unique_available_task')
    kind = detail['bound_kind']
    if kind == 'active_task':
        if context['execution_state'] != 'running' or not context.get('active_request_id'):
            detail.update(bound_kind='unresolved',status='active_task_unavailable')
        else:
            detail.update(request_id=context['active_request_id'],state_revision=context['state_revision'])
            detail['required_checks'].append('active_task_and_state_still_current')
    elif kind in ('previous_task','previous_program'):
        previous = context.get(kind)
        if previous is None or previous['status'] != 'unique':
            detail.update(bound_kind='unresolved',status='previous_reference_not_unique')
        else:
            detail.update(reference_id=previous['reference_id'],reference_revision=previous['revision'])
            detail['required_checks'].append('previous_reference_still_current')
    elif kind == 'prior_steps':
        detail['step_indices'] = list(ref['step_indices'])
    if detail['bound_kind'] == 'learned_memory':
        detail['required_checks'].append('learned_memory_lookup_and_state_check')
    if detail['bound_kind'] == 'previous_task' and op == 'continue':
        detail['required_checks'] += ['stop_confirmed','current_state_verified','replan_before_resume']
    return detail


def _range_violations(index, step):
    args = step.get('args', {}); violations=[]
    def check(field, value, low, high):
        if value < low or value > high:
            violations.append({'step_index':index,'field':field,'value':value,'supported':[low,high]})
    for field in ('source_slot','destination_slot','object_id','count'):
        if field in args: check(field,args[field],1,4)
    if 'height' in args: check('height',args['height'],1,2)
    if 'repeat_count' in args and args['repeat_count'] < 1:
        violations.append({'step_index':index,'field':'repeat_count','value':args['repeat_count'],'supported':'positive integer'})
    for row_index, row in enumerate(args.get('layout', [])):
        check(f'layout[{row_index}].slot',row['slot'],1,4)
        if 'count' in row: check(f'layout[{row_index}].count',row['count'],0,2)
        for obj_index,obj in enumerate(row.get('object_ids', [])):
            check(f'layout[{row_index}].object_ids[{obj_index}]',obj,1,4)
        if len(row.get('object_ids', []))>2:
            violations.append({'step_index':index,'field':f'layout[{row_index}].object_ids',
                               'value':len(row['object_ids']),'supported':[0,2]})
    return violations


def _v2_task(step):
    args=step.get('args', {}); op=step['op']
    layout = args.get('layout')
    if layout is not None:
        layout = [{'slot':row['slot'], 'count':row.get('count'),
                   'object_ids':copy.deepcopy(row.get('object_ids'))} for row in layout]
    number=lambda field: args.get(field)
    return {'op':'task','kind':op,'source_slot':number('source_slot'),
            'destination_slot':number('destination_slot'),'object_id':number('object_id'),
            'count':number('count'),'height':number('height'),
            'layout':layout,'evidence':step['evidence']}


def resolve_sketch(sketch, transcript, context, *, receipt_state_revision=None):
    """Build an auditable proposal from the same receipt snapshot, without execution.

    The returned v2 candidate is a diagnostic conversion, never an approval.
    Missing values remain absent/null; out-of-range values never become actions.
    """
    sketch=validate_sketch(sketch,transcript)
    try: normalized_context=v1.normalize_context(context)
    except v1.ProgramError as exc: raise SketchError(exc.code,str(exc)) from exc
    result={'schema':'intent_sketch_resolution.v1','raw_sketch':sketch,'proposal_only':True,
            'execution_authorized':False,'receipt_state_revision':normalized_context['state_revision'],
            'state_status':'receipt_snapshot_only','bindings':[],'excluded_steps':[],
            'range_violations':[],'issues':[],'required_checks':[],'independent_stop_diagnostic':[],
            'v2_candidate':None,'v2_conversion':'not_attempted','disposition':'unresolved'}
    if receipt_state_revision is not None and receipt_state_revision != normalized_context['state_revision']:
        result.update(state_status='state_changed',disposition='state_changed')
        result['required_checks'].append('obtain_new_receipt_context')
        return result
    requested=[]
    for index, step in enumerate(sketch['steps']):
        if step['polarity']=='exclude':
            result['excluded_steps'].append({'index':index,'step':copy.deepcopy(step)})
        elif step['polarity']=='request':
            requested.append((index,step))
            result['range_violations'] += _range_violations(index,step)
    for index,requested_step in requested:
        if requested_step['op']!='stop_motion':continue
        candidate={'schema':'task_program.v2','steps':[{'op':'stop_motion','scope':'active_task','evidence':requested_step['evidence']}]}
        for diagnostic in v2._validated_initial_stop(candidate,transcript,normalized_context,'diagnostic-stop'):
            result['independent_stop_diagnostic'].append({'sketch_step_index':index,**diagnostic})
    if result['range_violations']:
        result.update(disposition='unsupported_range',v2_conversion='not_representable_without_loss')
        return result
    converted=[]; mapping={}; prior_stop=False
    for index, step in enumerate(sketch['steps']):
        polarity,op,args=step['polarity'],step['op'],step.get('args', {})
        if polarity=='exclude': continue
        if polarity=='query' or op=='question':
            converted.append({'op':'clarify','reason':'question_only','question':None,'evidence':step['evidence']})
            continue
        if polarity=='uncertain' or op=='clarify':
            reason=args.get('reason')
            if reason is None:
                result['issues'].append({'step_index':index,'code':'missing_clarification_reason'})
                continue
            converted.append({'op':'clarify','reason':reason,'question':args.get('question'),'evidence':step['evidence']})
            continue
        if op in TASKS:
            if op in TRANSFER and ('destination_slot' not in args or not ({'source_slot','object_id'} & set(args))):
                result['issues'].append({'step_index':index,'code':'missing_target'})
                continue
            if op == 'swap' and not {'source_slot','destination_slot'} <= set(args):
                result['issues'].append({'step_index':index,'code':'missing_target'})
                continue
            if op == 'arrange' and 'layout' not in args:
                result['issues'].append({'step_index':index,'code':'missing_layout'})
                continue
            if op in TRANSFER | {'swap'} and 'source_slot' in args and 'destination_slot' in args and _same_value(args['source_slot'],args['destination_slot']):
                result['issues'].append({'step_index':index,'code':'same_source_and_destination'})
                continue
            if 'layout' in args and any(not ({'count','object_ids'} & set(x)) for x in args['layout']):
                result['issues'].append({'step_index':index,'code':'missing_layout_contents'})
                continue
            mapping[index]=len(converted); converted.append(_v2_task(step)); continue
        if op == 'stop_motion':
            mapping[index]=len(converted); converted.append({'op':'stop_motion','scope':'active_task','evidence':step['evidence']})
            prior_stop=True; continue
        if op == 'stop_repeat':
            if 'boundary' not in args:
                result['issues'].append({'step_index':index,'code':'missing_repeat_boundary'});continue
            mapping[index]=len(converted)
            converted.append({'op':'stop_repeat','boundary':args['boundary'],'evidence':step['evidence']}); continue
        if op == 'status':
            if 'subject' not in args:
                result['issues'].append({'step_index':index,'code':'missing_status_subject'});continue
            mapping[index]=len(converted)
            converted.append({'op':'status','subject':args['subject'],'evidence':step['evidence']}); continue
        if op in REFERENCE_KINDS:
            binding=_bind_reference(op,step['reference'],normalized_context)
            binding['step_index']=index; result['bindings'].append(binding)
            for check in binding['required_checks']:
                if check not in result['required_checks']:result['required_checks'].append(check)
            kind=binding['bound_kind']
            reference={'kind':kind,'reference':step['reference']['mention']}
            if op=='repeat':
                if kind=='prior_steps':
                    indices=step['reference']['step_indices']
                    if any(i not in mapping for i in indices):
                        result['issues'].append({'step_index':index,'code':'prior_step_unavailable'});continue
                    reference['step_indices']=[mapping[i] for i in indices]
                else:reference['step_indices']=None
                count={'mode':args['repeat_mode'],'value':args.get('repeat_count')}
                item={'op':'repeat','target':reference,'count':count,'evidence':step['evidence']}
                for check in ('repeat_plan_checked','current_state_checked_before_each_round'):
                    if check not in result['required_checks']:result['required_checks'].append(check)
                if count['mode']=='additional' and 'existing_repeat_plan_verified_or_clarified' not in result['required_checks']:
                    result['required_checks'].append('existing_repeat_plan_verified_or_clarified')
                if prior_stop:
                    for check in ('stop_confirmed','repeat_target_revalidated_after_stop'):
                        if check not in result['required_checks']:result['required_checks'].append(check)
            else:
                item={'op':op,'target':reference,'evidence':step['evidence']}
                if prior_stop and op=='return':
                    for check in ('stop_confirmed','current_state_verified_before_return'):
                        if check not in result['required_checks']:result['required_checks'].append(check)
                if prior_stop and op=='continue' and kind=='active_task':
                    result['issues'].append({'step_index':index,'code':'continue_after_stop_requires_replan'});continue
            mapping[index]=len(converted);converted.append(item);continue
        raise SketchError('invalid_operation', f'unhandled step {index}: {op}')
    if result['issues']:
        result.update(disposition='incomplete',v2_conversion='not_representable_without_loss')
        return result
    if not converted:
        # This reports the absence of a positive action, without manufacturing one.
        result.update(disposition='negated_only' if result['excluded_steps'] else 'no_requested_action',
                      v2_conversion='no_action_to_convert')
        return result
    if len(converted)>8:
        # v2 has an eight-record limit even for its own clarify/status records.
        # Preserve the entire sketch; never drop a question or requested step.
        result.update(disposition='v2_capacity_exceeded',v2_conversion='not_representable_without_loss')
        return result
    result['v2_candidate']={'schema':'task_program.v2','steps':converted}
    result['v2_conversion']='structural_conversion_only'
    result['disposition']='unresolved' if any(
        x['op']=='clarify' or (x['op'] in ('repeat','return','continue') and x['target']['kind']=='unresolved') or
        (x['op']=='repeat' and x['count']['mode']=='unresolved') or
        (x['op']=='stop_repeat' and x['boundary']=='unresolved') for x in converted) else 'proposal'
    return result


def validate_v2_candidate(resolution, transcript, context, request_id):
    """Apply the unchanged v2 guard to a losslessly convertible proposal only."""
    candidate=resolution.get('v2_candidate')
    if candidate is None:
        return {'applicable':False,'status':'not_representable','program':None,'detail':resolution['v2_conversion'],
                'validated_stop_prefix':[]}
    if not v1._valid_id(request_id):
        return {'applicable':False,'status':'invalid_request_id','program':None,'detail':'invalid caller-owned ID',
                'validated_stop_prefix':[]}
    program={**copy.deepcopy(candidate),'request_id':request_id}
    stops=[]
    def keep_stop(index,step):
        if step['op']=='stop_motion':stops.append({'step_index':index,'step':copy.deepcopy(step)})
    try:
        v2.validate_program(program,transcript,context,on_valid_step=keep_stop)
    except v1.ProgramError as exc:
        return {'applicable':True,'status':exc.code,'program':None,'detail':str(exc),'validated_stop_prefix':stops}
    return {'applicable':True,'status':'ok','program':program,'detail':None,'validated_stop_prefix':stops}
