#!/usr/bin/env python3
"""
CannyForge Streamlit Dashboard

Live view of rules, error rates, success rates, and learning history.

Usage:
    cannyforge dashboard
    # or: streamlit run cannyforge/dashboard.py

Requires: pip install cannyforge[dashboard]
"""

import streamlit as st
from cannyforge import CannyForge
from cannyforge.knowledge import RuleStatus


def main():
    st.set_page_config(page_title="CannyForge Dashboard", layout="wide")

    st.title("CannyForge Dashboard")
    st.markdown("Real-time view of your self-improving agent system")

    # Initialize Forge
    if 'forge' not in st.session_state:
        st.session_state.forge = CannyForge()

    forge = st.session_state.forge

    # Manual refresh button
    if st.button("Refresh"):
        st.rerun()

    # Get stats
    stats = forge.get_statistics()

    # Top row: key metrics
    col1, col2, col3, col4 = st.columns(4)

    exec_stats = stats["execution"]
    learn_stats = stats["learning"]
    kb_stats = stats["knowledge"]

    with col1:
        st.metric("Tasks Executed", exec_stats["tasks_executed"])
    with col2:
        st.metric("Success Rate", f"{exec_stats['success_rate']:.1%}")
    with col3:
        st.metric("Learning Cycles", learn_stats["learning_cycles"])
    with col4:
        st.metric("Total Rules", learn_stats["total_rules"])

    st.divider()

    # Two columns
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rule Status")
        by_status = kb_stats.get("rules_by_status", {})

        if by_status:
            # Use Streamlit's built-in bar_chart (no extra dependencies)
            chart_data = {
                s: by_status.get(s, 0) for s in ["active", "probation", "dormant"]
            }
            st.bar_chart(chart_data)
        else:
            st.info("No rules yet. Run some tasks to trigger learning!")

    with col_right:
        st.subheader("Skill Performance")

        skill_stats = stats["skills"]["skill_stats"]
        if skill_stats:
            for name, sdata in skill_stats.items():
                rate = sdata.get("success_rate", 0)
                executions = sdata.get("executions", 0)
                st.write(f"**{name}**: {rate:.0%} success ({executions} runs)")
        else:
            st.info("No executions yet")

    st.divider()

    # Rules table
    st.subheader("All Rules")

    skill_filter = st.selectbox(
        "Filter by skill",
        ["all"] + forge.skill_registry.list_skills()
    )

    if skill_filter == "all":
        skills = forge.skill_registry.list_skills()
    else:
        skills = [skill_filter]

    rules_data = []
    for skill_name in skills:
        rules = forge.knowledge_base.get_rules(skill_name)
        for rule in rules:
            rules_data.append({
                "Skill": skill_name,
                "Name": rule.name,
                "Type": rule.rule_type.value,
                "Status": rule.status.value,
                "Confidence": f"{rule.confidence:.2f}",
                "Effectiveness": f"{rule.effectiveness:.2f}",
                "Applied": rule.times_applied,
            })

    if rules_data:
        # Use Streamlit's native table (no pandas dependency)
        st.table(rules_data)
    else:
        st.info("No rules to display")

    st.divider()

    # Recent errors
    st.subheader("Recent Errors")

    errors = forge.learning_engine.error_repo.get_recent(10)
    if errors:
        for err in reversed(errors):
            st.write(f"- **{err.error_type}** in {err.skill_name}: {err.task_description[:50]}...")
    else:
        st.info("No errors recorded")


if __name__ == "__main__":
    main()
