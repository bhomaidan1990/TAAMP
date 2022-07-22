import copy

from pddlstream.algorithms.downward import fact_from_fd
from pddlstream.algorithms.reorder import get_partial_orders
from pddlstream.language.conversion import pddl_from_object
from pddlstream.language.object import OptimisticObject, UniqueOptValue
from pddlstream.language.function import FunctionResult
from pddlstream.utils import neighbors_from_orders, get_mapping, safe_zip

def get_steps_from_stream(stream_plan, step_from_fact, node_from_atom):
    steps_from_stream = {}
    for result in reversed(stream_plan):
        steps_from_stream[result] = set()
        for fact in result.get_certified():
            if (fact in step_from_fact) and (node_from_atom[fact].result == result):
                steps_from_stream[result].update(step_from_fact[fact])
        for fact in result.instance.get_domain():
            step_from_fact[fact] = step_from_fact.get(fact, set()) | steps_from_stream[result]
            # TODO: apply this recursively
    return steps_from_stream

def get_fluent_instance(external, input_objects, state, real_states=[]):
    import pddl
    fluent_facts = map(fact_from_fd, filter(
        lambda f: isinstance(f, pddl.Atom) and (f.predicate in external.fluents), state))
    #atpose_fluent_facts = map(fact_from_fd, filter(
        #lambda f: isinstance(f, pddl.Atom) and (f.predicate in ["atpose"]), state))
    if len(real_states) == 0:
        real_states = [state]
    return external.get_instance(input_objects, fluent_facts=fluent_facts, atpose_fluent_facts=get_atposefluents(real_states))

def get_atposefluent_instance(external, input_objects, real_states):
    atpose_fluent_facts = get_atposefluent(real_states)

    instances = []
    for fact in atpose_fluent_facts:
        instances.append(external.get_instance(input_objects, atpose_fluent_facts=fact))

    return instances

def get_atposefluents(real_states):
    import pddl
    all_fluent_facts = []
    all_fluent_facts_length = []
    
    for state in real_states:
        fluent_facts = map(fact_from_fd, filter(
            lambda f: isinstance(f, pddl.Atom) and (f.predicate in ["atpose"]), state))
        all_fluent_facts.append(fluent_facts)
        all_fluent_facts_length.append(len(fluent_facts))
    
    max_length = max(all_fluent_facts_length)
    
    facts_to_return = []
    facts_strs = []
    for i in range(len(all_fluent_facts_length)):
        if all_fluent_facts_length[i] == max_length:
            fact_str = str(all_fluent_facts[i])
            if fact_str not in facts_strs:
                facts_to_return.append(all_fluent_facts[i])
                facts_strs.append(fact_str)
    
    #print "atpose fluent generated:"
    #for fact in facts_to_return:
        #print "\t", fact
    
    return facts_to_return

def convert_fluent_streams(stream_plan, real_states, action_plan, step_from_fact, node_from_atom):
    #return stream_plan
    import pddl
    assert len(real_states) == len(action_plan) + 1
    steps_from_stream = get_steps_from_stream(stream_plan, step_from_fact, node_from_atom)

    #print "[Meiying::apply_fluents::convert_fluent_streams] steps_from_stream:"
    #print steps_from_stream
    
    #print "[Meiying::convert_fluent_streams] real_states"
    #for each_state in real_states:
        #print "\t", each_state

    # TODO: ensure that derived facts aren't in fluents?
    # TODO: handle case where costs depend on the outputs
    _, outgoing_edges = neighbors_from_orders(get_partial_orders(stream_plan, init_facts=map(
        fact_from_fd, filter(lambda f: isinstance(f, pddl.Atom), real_states[0]))))
    static_plan = []
    fluent_plan = []
    for result in stream_plan:
        #print "[Meiying::apply_fluents::convert_fluent_streams] result:", result.external.name
        external = result.external
        #if result.external.name == "test-pick-feasible":
            #print "[Meiying::apply_fluents::convert_fluent_streams] FOUND test-pick-feasible. it's class: ", result.external.__class__ 
            #print "[Meiying::apply_fluents::convert_fluent_streams] FOUND test-pick-feasible. is it a test?: ", external.is_test
        if isinstance(result, FunctionResult) or (result.opt_index != 0) or (not external.is_fluent):
            #static_new_instance = get_atposefluent_instance(external, result.input_objects, real_states)[0]
            #for fact in get_atposefluents(real_states):
                #print "\t", fact
            if result.external.name in ["sample-pose", "sample-tool-goal", "sample-pull-tool-goal", "sample-push-tool-goal"]:
                new_instance = external.get_instance(result.instance.input_objects, fluent_facts=[], atpose_fluent_facts=get_atposefluents(real_states))     
                new_result = new_instance.get_result(result.output_objects, opt_index=result.opt_index)
                static_plan.append(new_result)
                #for fact in get_atposefluents(real_states):
                    #new_instance = external.get_instance(result.instance.input_objects, fluent_facts=[], atpose_fluent_facts=fact)     
                    #new_result = new_instance.get_result(result.output_objects, opt_index=result.opt_index)
                    #static_plan.append(new_result)
                #raise Exception("stop!!!")
                #for fact in get_atposefluents(real_states):
                    #new_instance = external.get_instance(result.instance.input_objects, fluent_facts=[], atpose_fluent_facts=atpose_fluent_facts)
                    #new_result = copy.deepcopy(result)
                    #static_plan.append(new_result)
                #result.instance.atpose_fluent_facts = get_atposefluents(real_states)[0]
                #static_plan.append(result)
                    #new_result = copy.deepcopy(result)
                    #new_result.instance.atpose_fluent_facts = fact
                    #static_plan.append(new_result)
                    #new_instance = external.get_instance(result.instance.input_objects, fluent_facts=[], atpose_fluent_facts=atpose_fluent_facts)
                    #new_output_objects = [
                        ##OptimisticObject.from_opt(out.value, object())
                        #OptimisticObject.from_opt(out.value, UniqueOptValue(result.instance, object(), name))
                        #for name, out in safe_zip(result.external.outputs, result.output_objects)]                    
                    #new_result = new_instance.get_result(new_output_objects, opt_index=result.opt_index)
                    #static_plan.append(new_result)
            else:
                static_plan.append(result)
            #result.instance.atpose_fluent_facts = get_atposefluents(real_states)[0]
            #new_output_objects = [
                ##OptimisticObject.from_opt(out.value, object())
                #OptimisticObject.from_opt(out.value, UniqueOptValue(result.instance, object(), name))
                #for name, out in safe_zip(result.external.outputs, result.output_objects)]
            #new_result = new_instance.get_result(new_output_objects, opt_index=result.opt_index)
            
            #static_plan.append(result) revert this
            #static_plan.append(new_result)
            
            #if result.external.name == "test-pick-feasible":
                #print "[Meiying::apply_fluents::convert_fluent_streams] ADD test-pick-feasible to static plan"
            continue
        if outgoing_edges[result]:
            # No way of taking into account the binding of fluent inputs when preventing cycles
            raise NotImplementedError('Fluent stream is required for another stream: {}'.format(result))
        #if (len(steps_from_stream[result]) != 1) and result.output_objects:
        #    raise NotImplementedError('Fluent stream required in multiple states: {}'.format(result))
        for state_index in steps_from_stream[result]:
            #if result.external.name == "test-pick-feasible":
                #print "[Meiying::apply_fluents::convert_fluent_streams] state_index:", state_index
            new_output_objects = [
                #OptimisticObject.from_opt(out.value, object())
                OptimisticObject.from_opt(out.value, UniqueOptValue(result.instance, object(), name))
                for name, out in safe_zip(result.external.outputs, result.output_objects)]
            if new_output_objects and (state_index <= len(action_plan) - 1):
                # TODO: check that the objects aren't used in any effects
                instance = copy.copy(action_plan[state_index])
                action_plan[state_index] = instance
                output_mapping = get_mapping(list(map(pddl_from_object, result.output_objects)),
                                             list(map(pddl_from_object, new_output_objects)))
                instance.var_mapping = {p: output_mapping.get(v, v)
                                        for p, v in instance.var_mapping.items()}
            new_instance = get_fluent_instance(external, result.instance.input_objects, real_states[state_index], real_states)
            # TODO: handle optimistic here
            new_result = new_instance.get_result(new_output_objects, opt_index=result.opt_index)
            #if result.external.name == "test-pick-feasible":
                #print "[Meiying::apply_fluents::convert_fluent_streams] new_result:", new_result
            fluent_plan.append(new_result)
    #print "[Meiying::apply_fluents::convert_fluent_streams] static_plan:"
    #print static_plan
    #print "[Meiying::apply_fluents::convert_fluent_streams] fluent_plan:"
    #print fluent_plan
    return static_plan + fluent_plan
