---
title: "Optimal patient physician matching"
excerpt: "An online medical platform is looking to have an optimal scheduling for patient assignment to physicians such that the number of patients who cannot find a physician meeting their criteria is minimized. At the same time the company looks for increasing their physicians utilization such that the number of medical experts who are not fully booked is minimized."
usemathjax: true
toc: true
toc_label: "Notebook table of content"
toc_sticky: true
author: Amir Nasrollahzadeh
categories:
  - White paper
tags:
  - optimization
  - matching
  - integer programming
---

An online medical platform is looking to have an optimal scheduling for patient assignment to physicians such that the number of patients who cannot find a physician meeting their criteria is minimized. At the same time the company looks for increasing their physicians utilization such that the number of medical experts who are not fully booked is minimized.

Assumptions {#se:assumptions}
===========

The following assumptions simplify the model formulation and notation
for the purposes of this white paper. Incorporating them into the model
will not be difficult and does not change the model significantly.
Assumptions in the list mentioned here are rather implicit. More
explicit assumptions are made in Section
[3](#se:var).

-   Each patient suffers only from one condition for each medical expert
    visit. Relaxing this assumption can be done in three phases: 1-
    patient is diagnosed with multiple conditions that can be treated in
    a single session (in each week) by a single expert. 2- patient is
    diagnosed with multiple conditions that cannot be treated in a
    single session per week by a single expert and requires multiple
    sessions. 3- patient is diagnosed with multiple conditions that
    require multiple experts per week. Incorporating the first
    relaxation is straightforward. Second and third relaxations can be
    implemented by careful construct of decision variables where
    multiple decision variables are created for the same patient.

-   The model does not take into consideration future demand of medical
    expertise and is focused on optimizing current utilization of
    experts schedules. It cannot add new medical experts to better serve
    patients and will not be able to remove experts to reduce cost.
    Furthermore, the model cannot anticipate future demands and will not
    be able to optimize its matching solution in w.r.t. to demand.
    Relaxing this assumption will probably require another modeling
    approach and a demand estimation model.

-   Once a patient and medical expert are assigned a schedule, time
    availability needs to be updated outside the scope of this model if
    the model needs to be re-solved for a new batch of patients. This
    assumption can also be relaxed by a process that sets decision
    variables according to the solution for the next batch of patients.

Decision variable {#se:var}
=================

The main decision variable is binary according to the following

$$X_{i,j,t} = \left\{
        \begin{array}{ll}
            1 & \text{if patient } i \text{ is assigned to expert } j \text{ at time } t,\\
            0 & \text{otherwise}
        \end{array}\right.$$

where $$j\in\mathcal{J}:=\{1, 2, \ldots, J\}$$ denotes the set of medical
experts, $$i\in\mathcal{I}:=\{1, 2, \ldots, I\}$$ denotes the set of new
patients, and $$t$$ denotes the appointment time slot in a week where
$$t\in\mathcal{T}:={t_1, t_2, \ldots, t}$$.

Medical expert {#sse:var-medic}
--------------

Each medical expert comes from a certain demographics which will be
denoted by $$d_j$$ where
$$d_j\in\mathcal{D}^{\mathcal{J}}:=\{d_1, d_2, \ldots, d_J\}$$ and $$d_j$$
might cover demographical factors such as
$$d_j= (\text{gender, ethnicity, ...})$$. Each medical expert might be
able to only treat certain conditions which will be denoted by
$$c_j = \{c^1_j, c^2_j, \ldots \}$$ and can provide certain services like
consulting, therapy, prescription which will be showed by
$$s_j = \{s^1_j, s^2_j, \ldots \}$$. Medical experts are assumed to be
licensed to practice in some locations (states) which is denoted by
$$l_j=\{l^1_j, l^2_j, \ldots\}$$ and are covered in network by certain
insurance providers, i.e., $$p_j = \{p^1_j, p^2_j, \ldots\}$$. Finally, it
is assumed that all medical experts have an availability schedule that
is capture by $$t_j = \{t^1_j, t^2_j, \ldots\}$$ for each week.

Patient {#sse:var-patient}
-------

Associated with each patient, are patient's preference for the medical
expert's demographics which will be denoted by $$d^i$$, patient's
condition $$c^i$$, patient's required services
$$s^i = \{s^1_i, s^2_i, \ldots\}$$, location $$l^i$$, insurance provider
$$p^i$$, and a set of available times $$t^i = \{t^1_i, t^2_i, \ldots\}$$.

Potential business rules {#se:biz-rule}
========================

These section reflects *our* understanding of some of the potential
business rules. These constraint can be modified to reflect current
business rules more accurately.

1.  A patient is matched to an expert only if the expert's demographics
    matches patients preferences.

    $$\begin{array}{ll}
                d^i - d_j \leq M  (1 - X_{i,j,t}) & \forall i\in\mathcal{I}, \forall j \in\mathcal{J}, \forall t\in\mathcal{T}\\
                d_j - d^i \leq M  (1 - X_{i,j,t}) & \forall i\in\mathcal{I}, \forall j \in\mathcal{J}, \forall t\in\mathcal{T}
            \end{array}$$

    where $$M$$ is a very large number. These set of constraints allow $$X_{i,j,t}$$ to be $$1$$ only if $$d^i = d_j$$.

2.  An expert is matched to a patient only if the expert can treat the
    patient's diagnosed condition.

    $$\begin{array}{ll}
                c^i - c^k_j \leq M  (1 - Y^k_j) & \forall k=1,\ldots,|c_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}\\
                c^k_j - c^i \leq M  (1 - Y^k_j) & \forall k=1,\ldots,|c_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}\\
                X_{i,j,t} \leq \sum_{k} Y^k_j & \forall k=1,\ldots,|c_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}, \forall t\in\mathcal{T}
            \end{array}$$

    where $$Y^k_j\in\{0, 1\}$$. These set of constraints allow $$X_{i,j,t}$$ to be $$1$$ only when one of the $$Y^k_j$$
    is equal to $$1$$ which means that $$c^i$$ has matched with one member
    of $$c_j$$, i.e., $$c^i\in c_j$$.

3.  A patient is matched to an expert only if the expert can provide the
    services the patient might require

    $$\begin{array}{ll}
                s^i_m - s^n_j \leq M (1 - Z^{i, m}_{n, j}) & \forall m=1,\ldots,|s^i|, \forall n=1,\ldots,|s_j|,\forall i\in\mathcal{I}, \forall j\in\mathcal{J}\\
                s^n_j - s^i_m \leq M (1 - Z^{i, m}_{n, j}) & \forall m=1,\ldots,|s^i|, \forall n=1,\ldots,|s_j|,\forall i\in\mathcal{I}, \forall j\in\mathcal{J}\\
                |s^i| X_{i,j,t} \leq \sum_n \sum_m Z^{i, m}_{n, j} & \forall m=1,\ldots,|s^i|, \forall n=1,\ldots,|s_j|,\forall i\in\mathcal{I}, \forall j\in\mathcal{J}, \forall t\in\mathcal{T}
            \end{array}$$

    where $$Z^{i, m}_{n, j}\in\{0, 1\}$$. These set of constraint allow $$X_{i,j,t}$$ to be $$1$$ only when $$|s^i|$$ of
    $$Z^{i, m}_{n, j}\in\{0, 1\}$$ are equal to $$1$$. In other words,
    $$s^i\subseteq s_j$$.

4.  An expert is matched to a patient only if the expert is licensed to
    practice in patient's location

    $$\begin{array}{ll}
                l^i - l^k_j \leq M  (1 - W^k_j) & \forall k=1,\ldots,|l_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}\\
                l^k_j - l^i \leq M  (1 - W^k_j) & \forall k=1,\ldots,|l_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}\\
                X_{i,j,t} \leq \sum_{k} W^k_j & \forall k=1,\ldots,|l_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}, \forall t\in\mathcal{T}
            \end{array}$$

    where $$W^k_j\in\{0, 1\}$$. These set of constraints allow $$X_{i,j,t}$$ to be $$1$$ only when one of the $$W^k_j$$
    is equal to $$1$$ which means that $$l^i$$ has matched with one member
    of $$l_j$$, i.e., $$l^i\in l_j$$.

5.  A patient is matched to an expert only if the expert is in patient's
    insurance network

    $$\begin{array}{ll}
                p^i - p^k_j \leq M  (1 - U^k_j) & \forall k=1,\ldots,|l_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}\\
                p^k_j - p^i \leq M  (1 - U^k_j) & \forall k=1,\ldots,|l_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}\\
                X_{i,j,t} \leq \sum_{k} U^k_j & \forall k=1,\ldots,|p_j|, \forall i\in\mathcal{I}, \forall j \in\mathcal{J}, \forall t\in\mathcal{T}
            \end{array}$$

    where $$U^k_j\in\{0, 1\}$$. These set of
    constraints allow $$X_{i,j,t}$$ to be $$1$$ only when one of the $$U^k_j$$
    is equal to $$1$$ which means that $$p^i$$ has matched with one member
    of $$p_j$$, i.e., $$p^i\in p_j$$.

6.  An expert is matched to a patient only if patient's available time
    matches the expert's free schedule.

    $$\begin{array}{ll}
                t^i_m - t^n_j \leq M (1 - V^{i, m}_{n, j}) & \forall m=1,\ldots,|t^i|, \forall n=1,\ldots,|t_j|,\forall i\in\mathcal{I}, \forall j\in\mathcal{J}\\
                t^n_j - t^i_m \leq M (1 - V^{i, m}_{n, j}) & \forall m=1,\ldots,|t^i|, \forall n=1,\ldots,|t_j|,\forall i\in\mathcal{I}, \forall j\in\mathcal{J}\\
                X_{i,j,t} \leq \sum_n \sum_m V^{i, m}_{n, j} & \forall m=1,\ldots,|t^i|, \forall n=1,\ldots,|t_j|,\forall i\in\mathcal{I}, \forall j\in\mathcal{J}, \forall t\in\mathcal{T}
            \end{array}$$

    where $$V^{i, m}_{n, j}\in\{0, 1\}$$. These set
    of constraint allow $$X_{i,j,t}$$ to be $$1$$ only when one of
    $$t^{i, m}_{n, j}\in\{0, 1\}$$ is equal to $$1$$. In other words,
    $$\exists\, t\in t^i; t\in t_j$$.

Other constraints {#sse:biz-cons}
-----------------

In addition to the business rules in Section
[4](#se:biz-rule){reference-type="ref" reference="se:biz-rule"}, this
section will include other constraint that ensure a feasible and
sensible solution to the optimization problem.

1.  Each patient is only assigned to (at most) one medical expert once
    in a period.

    $$\sum_{j}\sum_{t} X_{i,j,t} \leq 1 \qquad\qquad \forall i\in\mathcal{I}$$

    where $$t\in \mathcal{T}$$ and
    $$\mathcal{T} = \bigcup_{(j\in\mathcal{J})} t_j$$.

2.  In each time slot, each expert is assigned to at most one patient.

    $$\sum_i X_{i,j,t} \leq 1 \qquad\qquad \forall j\in\mathcal{J}, t\in\mathcal{T}$$

Business goals {#se:biz-goal}
==============

It is assumed that the main interests of the business are according to
the following:

1.  Minimize the number of new patients who are not assigned to a
    medical expert. This is the same as maximizing the number of
    patients who are assigned a schedule for treatment.

    $$\max \,\,\, \sum_{i\in\mathcal{I}} \sum_{j\in\mathcal{J}} \sum_{t\in\mathcal{T}} X_{i,j,t}$$

2.  Maximize the expert's utilization such that the number of experts
    that do not have a full schedule is minimized. Medical expert $j$ is
    not fully scheduled if
    $$|t_j| - \sum_{i\in\mathcal{I}}\sum_{t\in\mathcal{T}} X_{i,j,t} \geq 0$$.
    Minimizing such experts can be done according to the following
    formulation

    $$\begin{array}{rll}
                \min & \sum_{j\in\mathcal{J}} \lambda_j & \\
                \text{subject to} &  &\\
                & |t_j| - \sum_{i\in\mathcal{I}}\sum_{t\in\mathcal{T}} X_{i,j,t} \leq |t_j| \lambda_j & \forall j\in\mathcal{J}
            \end{array}$$

Objective setup {#sse:biz-obj}
---------------

To achieve the stated business goals, the objective function of the
overall optimization formulation can be set in different ways which will
provide different solutions, customizations, insights, and are at
different levels of complexity.

1.  Affine combination of both objectives where

    $$\max \,\,\, \alpha \Big(\sum_{i\in\mathcal{I}} \sum_{j\in\mathcal{J}} \sum_{t\in\mathcal{T}} X_{i,j,t}\Big) - \beta \Big(\sum_{j\in\mathcal{J}} \lambda_j\Big)$$

    denotes the overall objective function. $$\alpha$$ and $$\beta$$ denote
    the relative weight of each term in the objective. This setup is
    relatively easy and straightforward to implement. However, tuning
    $$\alpha$$ and $$\beta$$ will require some experimentation and input
    from business.

2.  Sequential optimization problems where

$$\begin{array}{rll}
                \max & \sum_{i\in\mathcal{I}} \sum_{j\in\mathcal{J}} \sum_{t\in\mathcal{T}} X_{i,j,t} & \\
                \text{subject to} & & \\
                & \text{constraints in Sections \ref{se:biz-rule} \& \ref{sse:biz-cons}} &
            \end{array}$$

    Now, assuming that the optimal solution to the
    above problem is $$x^*$$, we control the relaxation of this optimality
    as a constraint into another optimization problem where
    $$\begin{array}{rll}
                \min & \sum_{j\in\mathcal{J}} \lambda_j & \\
                \text{subject to} & & \\
                & \text{constraints in Sections \ref{se:biz-rule} \& \ref{sse:biz-cons}}, & \\
                & |t_j| - \sum_{i\in\mathcal{I}}\sum_{t\in\mathcal{T}} X_{i,j,t} \leq |t_j| \lambda_j, & \forall j\in\mathcal{J},\\
                & \sum_{i\in\mathcal{I}} \sum_{j\in\mathcal{J}} \sum_{t\in\mathcal{T}} X_{i,j,t} \geq (1 - \gamma) x^*.&
            \end{array}$$

    This sequence can be set up in reverse as
    well. $$\gamma\in[0,1)$$ relaxes the first objective. Typically,
    $$\gamma$$ is set to a number between $$1\%$$ to $$5\%$$.

3.  Multi-objective programming either by goal programming or by Pareto
    optimization. In goal programming, the order of objective should be
    known whereas in Pareto optimization the order is not required. Note
    that goal programming is considerably easier to solve than Pareto
    optimization.

    $$\begin{array}{rll}
                \max & \sum_{i\in\mathcal{I}} \sum_{j\in\mathcal{J}} \sum_{t\in\mathcal{T}} X_{i,j,t} & \\
                \min & \sum_{j\in\mathcal{J}} \lambda_j & \\
                \text{subject to} & & \\
                & \text{constraints in Sections \ref{se:biz-rule} \& \ref{sse:biz-cons}}, & \\
                & |t_j| - \sum_{i\in\mathcal{I}}\sum_{t\in\mathcal{T}} X_{i,j,t} \leq |t_j| \lambda_j, & \forall j\in\mathcal{J}
            \end{array}$$

Impact {#se:impact}
======
