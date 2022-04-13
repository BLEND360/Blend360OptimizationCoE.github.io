---
title: "Infeasibility resolution with PuLP"
excerpt: "Resolving infeasibility of large problems depends on identifying the conflicting constraints which is not a simple task. A problem could have multiple sources of infeasibility that may or may not overlap. Relaxing a problem's constraints always resolves the infeasibility but it would not identify the source necessarily since it always depends on how much penalty is assigned to each relaxation coefficient. The usual optimization solvers also do not have a perfect way of resolving infeasibility since the exact method of identifying the source of infeasibility and then determining the minimum number of constraints that have to be removed is a problem that can end up being more complex than the original problem."
usemathjax: true
toc: true
toc_label: "Notebook table of content"
toc_sticky: true
authors:
  - Amir Nasrollahzadeh
  - Aneesh Muthiyan
header:
  teaser: "/assets/images/Post-images/2022-04-11-infeasibility_resolution_with_pulp/three-possible-outcomes.png"
categories:
  - Solution notebook
tags:
  - optimization
  - infeasibility
  - linear programming
  - PuLP
---

In mathematical programming models, infeasibility refers to a situation where no solution satisfying all constraints could be found. In large problems, it is often the case that infeasibility is caused by multiple conflicting constraints. The source of infeasibility might be data problems that affect parameters, wrong decision variable bounds, or errors in writing constraints to capture a business rule. Whatever the source of infeasibility, resolving it depends on identifying the conflicting constraints.

An example of conflicting constraints is in the following:

$$
\begin{align*}
            \min  &  x_1 + x_2 + x_3     &\\
\text{subject to} &                      &\\
                  & x_1 + x_2 \leq 10,   &\\
                  & 2x_1 + 3x_3 \leq 5,  &\\
                  & x_3 - x_1 \geq 10,   &\\
                  & x_1, x_2, x_3 \geq 0 &
\end{align*}
$$

The above constraints do not allow for any feasible solution. In particular, since all decision variables are set to be greater or equal to $$0$$, constraint $$x_3 - x_1 \geq 10$$ results in $$x_3 \geq 10$$ which will make satisfying constraint $$2x_1 + 3x_3 \leq 5$$ impossible. If infeasibility is not induced by parameter, bound, or inequality misspecification, resolving infeasibility is only possible by removing conflicting constraints. However, one should note that in case of infeasibility, it is often the case that multiple (or all) constraints are at conflict with each other. In the example above, removing any of the following three constraints, $$x_3 - x_1 \geq 10$$, $$x_1 \geq 0$$, or $$2x_1 + 3x_3 \leq 5$$ resolves the infeasibility.

## 1- Irreducible Infeasible Set

For a mathematical programming problem, an irreducible infeasible set (IIS) is an infeasible subset of constraints that become feasible if any single constraint is removed. In the above example, the IIS is comprised of $$x_3 - x_1 \geq 10$$, $$x_1 \geq 0$$, and $$2x_1 + 3x_3 \leq 5$$. IISs are not unique and an infeasible problem may have multiple infeasible subsystems; for example, consider a case where a problem has multiple sources of infeasibility. A problem can have multiple overlapping or separate IISs.

The number of IISs can be exponential in the size of the original problem. Although, removing a constraint from each of these IISs will make the original problem feasible, not all conflicting constraint have the same effect on the original problem. For example, removing $$x_1 \geq 0$$ will make the original problem feasible but it would not make sense if $$x_1$$ is defined to capture a practical decision variable for which negative values is not acceptable.

For problems with multiple sources of infeasibility or with multiple separate or overlapping IISs, the minimum number of constraints that have to be dropped to make the original problem feasible is not as straightforward. Therefore, for practical debugging of infeasibility, it will be useful to find the min cover IISs which will produce the minimum number of constraints that have to be removed to make the original problem feasible. This is equivalent to a set covering problem of the following type

$$
\begin{align*}
            \min  & \,1^T y     &\\
\text{subject to} &                      &\\
                  & \sum\limits_{i\in S_j} y_i \geq 1,   & j=1,\ldots,r,\\
                  & y_i  \geq 0, & i=1,\ldots,m.
\end{align*}
$$

where $$S_j$$ is the set of indices of the inequalities in the $$j$$th IIS of a problem with $$r$$ IISs. Let $$y = (y_1,\ldots,y_m)$$ where $$y_i$$ is a binary variable whose value will be $$1$$ where constraint $$i$$ is chosen to be deleted, and $$0$$ otherwise.

For the example above, the only IIS is equal to $$\{x_3 - x_1 \geq 10, x_1 \geq 0, 2x_1 + 3x_3 \leq 5\}$$. Any other subset of constraints will not be infeasible. Therefore, the minimum number of constraints to be removed to make the original problem feasible is $$1$$. Note that in this case, because of a singular IIS, removing one constraint from the IIS will make both the IIS and the original problem feasible.

## 2- Maximum Cardinality Feasible Set

The maximum cardinality feasible set is equal to the largest subset of constraints that are feasible to the original problem. Removing the constraints identified by the above set covering problem will result in a maximum cardinality feasible set. However, the set covering problem may result in multiple optimal solutions which in turn will result in multiple maximum cardinality feasible sets. The decision-maker is then have to assess each maximal feasible set to determine which set better serves the decision model.

In this example, from the feasibility perspective, removing any of the constraints in the IIS will result in a different maximum cardinality feasible set.

$$
\begin{align*}
    &\text{Set }1       & \qquad &\text{Set }2         & \qquad &\text{Set }3\\
    &x_1 + x_2 \leq 10  & \qquad &x_1 + x_2 \leq 10    & \qquad &x_1 + x_2 \leq 10\\
    &2x_1 + 3x_3 \leq 5 & \qquad &2x_1 + 3x_3 \leq 5   & \qquad &x_3 - x_1 \geq 10 \\
    &x_3 - x_1 \geq 10  & \qquad &x_1, x_2, x_3 \geq 0 & \qquad &x_1, x_2, x_3 \geq 0\\
    &x_2, x_3 \geq 0    & \qquad &                     & \qquad &
\end{align*}
$$

Each of the above sets have the same number of constraints and are feasible. Each set has its own practical consequences for the solution which will inform the decision-maker's choice.

## 3- Infeasibility Resolution in Practice

### 3.1- Removing conflicting constraints

Some optimization engines have tools that help constraints conflict resolution by identifying a set of **suspicious** constraints that may cause infeasibility. In this Section, Gurobi, CPLEX, and PuLP's capability to resolve constraints are briefly discussed.

**Gurobi**: Gurobi's [Model.computeIIS()](https://www.gurobi.com/documentation/9.5/refman/py_model_computeiis.html) is able to calculate an IIS of the original problem. However, Gurobi is unable to recognize all IISs of a problem and cannot identify if the given IIS has minimum cardinality. Therefore, for problems with multiple sources of infeasibility, the IIS given by Gurobi will not be the ultimate solution. Note that whether a problem has one or more than one sources of infeasibility is not known in advance. Therefore, identifying all IISs and thus maximum cardinality feasible sets will be an iterative process with trial and error.

**CPLEX**: The [presolve](https://www.ibm.com/docs/en/icos/12.8.0.0?topic=performance-preprocessing) method in CPLEX relies on dual reduction of the original problem. If the presolve methods is infeasible, CPLEX has no way of showing conflicting constraints. However, if the presolve solution is available but the original problem is infeasible, CPLEX will be able to take advantage of its primal-dual Simplex method and reports a set of refined conflicts. These conflicts are more general than IISs and will also apply to mixed integer problems. Unlike the Gurobi optimizer, conflict refiner in CPLEX can report a set of minimal constraints and bounds that are contradictory [see here](https://www.ibm.com/docs/en/icos/12.8.0.0?topic=conflicts-how-conflict-differs-from-iis). However, this method still cannot identify all conflicts when the underlying problem has multiple sources of infeasibility.

**PuLP**: At the time of this writing, PuLP has no official way of conflict resolution documented. Although, it has functional capabilities to provide some level of diagnosis, almost on par with Gurobi but inferior to CPLEX.

### 3.2- Feasibility relaxation

Another heuristic way to resolve infeasibility will be to relax all bounds and constraints, penalize the violations in the objective function heavily, and solve the problem. The optimization force will try not to violate the constraints if the problem is feasible. In case of infeasibility, the decision-maker can tune the order and the amount of violation by setting the penalty terms.

Although this method will allow a relaxed feasible solution to the original problem, the constraints that end up being violated do not necessarily match the minimal IIS and thus all conflicting constraints may not be identified. In the above example, any of the constraints in the set $$\{x_3 - x_1 \geq 10, x_1 \geq 0, 2x_1 + 3x_3 \leq 5\}$$ maybe violated at a price. For example, $$x_1 = -10$$ violates the $$x_1 \geq 0$$ constraint and is a feasible solution to the other two which will prevent them from violation. Therefore, the decision maker is unable to recognize the other two constraints as conflicting.

CPLEX's [FeasOpt](https://www.ibm.com/docs/en/icos/12.8.0.0?topic=infeasibility-repairing-feasopt), Gurobi's [Model.feasRelax()](https://www.gurobi.com/documentation/9.1/refman/py_model_feasrelax.html), and PuLP's [elastic constraints](https://coin-or.github.io/pulp/guides/how_to_elastic_constraints.html) are different implementation of relaxation approaches in each solver. Note that while CPLEX and Gurobi have automated the process of relaxing all or some of the constraints, PuLP's elastic constraint method is applicable to only one constraint at a time.

## 4- Infeasibility Resolution with PuLP Helper Functions

While PuLP does not have an explicit method to identify conflicting constraints, a deep dive into its source code publicly available in Github ([here](https://github.com/coin-or/pulp/blob/master/pulp/pulp.py)) reveals helper functions that have proven to be very useful in conflict resolution. Class `LpConstraint` of the `pulp.py` module includes a `valid(eps = 0)` method which returns a Boolean determining whether a constraint is violated by more than epsilon `eps`. Similarly, class `LpVariable` also includes a `valid(eps = 0)` method which returns a Boolean indicating whether a decision variable has violated its upper and lower bounds by more than `eps`. These two methods may be used to construct a **suspicious** set of constraints (and variable bounds) to be refined into a minimal list of conflicting constraints.

### 4.1- Identify violated constraints and decision bounds

Given a PuLP object of type `LpProblem`, the functions `violated_constr` and `violated_var` identify which constraints and variables are violated. In both functions, all constraints and variables of the problem object are iterated over to check for violation. If violated, the name of the constraint or decision variable is returned. Note that to identify infeasibility to the original problem, `eps` is set to zero for both validation method.


```python
def violated_constr(prob):
    import pulp
    suspected_constr = []
    for c in prob.constraints.values():
        if not c.valid(0):
            ## check if the constraint is a soft constraint;
            ## soft constraints should not cause infeasibility and may be ignored
            constr_name = c.toDict()['name']
            suspected_constr.append(constr_name)
    return suspected_constr
```


```python
def violating_var(prob):
    import pulp
    suspected_var = []
    for v in prob.variables():
        if not v.valid(0):
            var_name = v.toDict()['name']
            suspected_var.append(var_name)
    return suspected_var
```

### 4.2- Removing violating constraints and relaxing decision bounds

Once a suspected list of constraints and decision variables are generated, the problem can be reduced (in terms of constraints) and relaxed (with respect to suspected decision variable bounds) and resolved to determine whether the infeasibility issue is fixed. To easily remove a constraint or manipulate a decision variable, the pulp object `LpProblem` is converted to a json file. To avoid issues with *numpy* numeric types, a json encoder is defined as the class `NpEncoder`.


```python
import json
import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
```

An `LpProblem` type can be read and written to a json object with the proper encoder. The rest of this infeasibility resolution guide will rely on a json instance of the optimization problem.


```python
prob.to_json(path_to_model, cls = NpEncoder)

def read_model_json(path_to_model):
    with open(path_to_model, "r") as input_file:
        prob_json = json.load(input_file)
    return prob_json
```

Function `rm_constr` removes suspected constraints of the `suspected_constr` list from the problem's json object in `prob_json`. This is done to a copy of the constraints from the original problem though. Function `relax_var` iterates over all decision variables and resets the decision bounds to infinity for variables in the `suspected_var` list. Similar to constraint, this process is also done to a copy of the decision variables from the original problem.


```python
def rem_constr(prob_json, suspected_constr):
    constr_lst = prob_json['constraints'].copy()
    for constr in prob_json['constraints']:
        if constr['name'] in suspected_constr:
            costr_lst.remove(constr)
    return costr_lst

def relax_var(prob_json, suspected_var):
    var_lst = prob_json['variables'].copy()
    for i, var in enumerate(prob_json['variables']):
        if var['name'] in suspected_var:
            var_lst[i]['lowBound'] = None
            var_lst[i]['upBound'] = None
    return var_lst
```

### 4.3- Maximum cardinality feasible set with PuLP

Below, a heuristic to find the maximum cardinality feasible set with PuLP is given. For this example, assume that infeasibility is only caused by explicit constraints (not decision bounds). A json instance of an LP problem is read in the algorithm below. The LP is run for the first time to get its optimality status. If the initial optimal status if `Infeasible`, a set of suspected constraints are generated by `violated_constr` function. This set is then removed from the constraint list of the original problem and the problem is resolved again. If `Infeasible` again, a new set is generated and appended to the previous one. If the reduced problem is `Optimal`, the suspected constraints are added back one by one. Each time, the new extended problem is resolved. If optimal, the constraint is kept and if infeasible, the constraint is marked as `unsafe` and is removed again from the extended problem. At the end of the `while` loop, a refined list of conflicting constraints should be produced. Note that this is a heuristic algorithm which, in case of multiple sources of infeasibility, will only result in one maximum cardinality feasible set. The method is also biased with respect to the order of constraints. However, for large problems, when the intermediate solution is achieved with a large number of feasible constraints, the algorithm works surprisingly well.

Also note that this is only possible if the underlying solution method to the LP is of the primal-dual Simplex type where feasible intermediate solutions are typically achieved for large infeasible problems. However, if the problem is infeasible from the presolve stage, the algorithm below will not be able to resolve infeasibility.


```python
orig_path_to_model = r"path to model's json"
suspected_constr = [pd.DataFrame(columns = cols)]

##run the lp for first time
var, prob = pulp.LpProblem.from_json(orig_path_to_model)
prob.solve()
print(f"status: {prob.status}, {pulp.LpStatus[prob.status]}")
print(f"objective: {prob.objective.value()}")
status = pulp.LpStatus[prob.status]
orig_prob_json = read_model_json(orig_path_to_model)
prob_json = orig_prob_json.copy()

while status == "Infeasible":
    ##create initial list of suspected constraints
    new_found_constr = violated_constr(prob)
    if new_found_constr.empty:
        print("Error! No more constraints are suspected but the infeasibility issue persists.")
        break
    suspected_constr = suspected_constr.extend(new_found_constr)

    ##remove identified violating hard constraints
    print(f"Unique number of new suspect constraints found: {len(new_found_constr)}")
    print(f"Identifying hard constraints to be removed from the original problem...")    
    reduced_constr_lst = rem_constr(prob_json, suspected_constr)

    ##resolve the problem after constraint removal
    print(f"Solving the reduced problem...")
    prob_json['constraints'] = reduced_constr_lst
    var, prob = pulp.LpProblem.from_dict(prob_json)
    prob.solve()
    print(f"status: {prob.status}, {pulp.LpStatus[prob.status]}")
    print(f"objective: {prob.objective.value()}")
    status = pulp.LpStatus[prob.status]

    ##refine the list of suspect constraints
    if pulp.LpStatus[prob.status] == "Optimal":
        safe_constr = []
        unsafe_constr = []
        print("Add back suspected hard constraints one by one...")
        for constr in suspected_constr:
            constr_lst.extend(constr)
            prob_json['constraints'] = constr_lst
            new_var, new_prob = pulp.LpProblem.from_dict(prob_json)
            new_prob.solve()
            if pulp.LpStatus[new_prob.status] == "Optimal":
                safe_constr.append(constr)
                print(f"{constr} from the suspected list is safe")
            else:
                unsafe_constr.append(constr)
                print(f"{constr} from the suspected list is conflicting")
                constr_lst.remove(constr)
        print(f"List of conflicting constr: {unsafe_constr}")
```

Furthermore, for this example, it was assumed that the infeasibility is only caused by conflicting constraints. In a general case, decision bounds may be one of the culprits too. In order to incorporate decision bound relaxation, the algorithm should treat decision bounds as constraints. Having reset all suspected decision bounds to infinity at first, each original decision bound is then applied one by one in a process very similar to the algorithm described above.
