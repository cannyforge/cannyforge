#!/usr/bin/env python3
"""
CannyForge Streamlit Dashboard

Live view of rules, corrections, error rates, success rates, and learning history.

Usage:
    cannyforge dashboard
    # or: streamlit run cannyforge/dashboard.py

Requires: pip install cannyforge[dashboard]
"""

import streamlit as st
from cannyforge import CannyForge
from cannyforge.knowledge import RuleStatus


def _eff_label(times_injected: int, times_effective: int) -> str:
    if times_injected == 0:
        return "unrated"
    return f"{times_effective / times_injected:.0%}"


def main():
    st.set_page_config(page_title="CannyForge Dashboard", layout="wide")

    st.title("CannyForge Dashboard")
    st.markdown("Real-time view of your self-improving agent system")

    # Initialize Forge
    if 'forge' not in st.session_state:
        st.session_state.forge = CannyForge()

    forge = st.session_state.forge

    if st.button("Refresh"):
        st.session_state.forge = CannyForge()
        st.rerun()

    # Get stats
    stats = forge.get_statistics()

    exec_stats = stats["execution"]
    learn_stats = stats["learning"]
    kb_stats = stats["knowledge"]

    # Top row: key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Tasks Executed", exec_stats["tasks_executed"])
    with col2:
        st.metric("Success Rate", f"{exec_stats['success_rate']:.1%}")
    with col3:
        st.metric("Learning Cycles", learn_stats["learning_cycles"])
    with col4:
        st.metric("Rules", learn_stats["total_rules"])
    with col5:
        st.metric("Corrections", kb_stats.get("total_corrections", 0))

    st.divider()

    # ── Corrections ────────────────────────────────────────────────────────────
    st.subheader("Corrections")

    # Collect all corrections across all skills known to the knowledge base
    all_skills_with_corrections = list(forge.knowledge_base.corrections_by_skill.keys())
    corrections_data = []
    for skill_name in all_skills_with_corrections:
        for c in forge.knowledge_base.get_corrections(skill_name):
            eff = _eff_label(c.times_injected, c.times_effective)
            corrections_data.append({
                "Skill": skill_name,
                "ID": c.id,
                "Type": c.correction_type or c.error_type,
                "Injected": c.times_injected,
                "Effective": c.times_effective,
                "Effectiveness": eff,
                "Content": c.content,
            })

    if corrections_data:
        # Summary bar: corrections per skill
        col_chart, col_detail = st.columns([1, 2])

        with col_chart:
            st.caption("Corrections by skill")
            by_skill_count = {}
            for row in corrections_data:
                by_skill_count[row["Skill"]] = by_skill_count.get(row["Skill"], 0) + 1
            st.bar_chart(by_skill_count)

        with col_detail:
            st.caption("Injection statistics")
            summary = [
                {
                    "Skill": row["Skill"],
                    "Type": row["Type"],
                    "Injected": row["Injected"],
                    "Effective": row["Effective"],
                    "Effectiveness": row["Effectiveness"],
                }
                for row in corrections_data
            ]
            st.table(summary)

        st.caption("Correction content")
        skill_options = ["all"] + sorted(set(r["Skill"] for r in corrections_data))
        skill_filter_c = st.selectbox("Filter corrections by skill", skill_options,
                                      key="corrections_skill_filter")
        filtered = corrections_data if skill_filter_c == "all" else [
            r for r in corrections_data if r["Skill"] == skill_filter_c
        ]
        for row in filtered:
            with st.expander(f"[{row['Skill']}] {row['Type']} · {row['ID']} · {row['Effectiveness']}"):
                st.write(row["Content"])
    else:
        st.info("No corrections loaded. Import a bundle with `cannyforge import <bundle.cannyforge> --skill <name>`.")

    st.divider()

    # ── Rules + Skill Performance ──────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rule Status")
        by_status = kb_stats.get("rules_by_status", {})
        if any(by_status.get(s, 0) > 0 for s in ["active", "probation", "dormant"]):
            chart_data = {s: by_status.get(s, 0) for s in ["active", "probation", "dormant"]}
            st.bar_chart(chart_data)
        else:
            st.info("No rules yet. Run tasks to trigger learning.")

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

    # ── Rules table ────────────────────────────────────────────────────────────
    st.subheader("Rules")

    skill_filter_r = st.selectbox(
        "Filter rules by skill",
        ["all"] + forge.skill_registry.list_skills(),
        key="rules_skill_filter",
    )
    rule_skills = forge.skill_registry.list_skills() if skill_filter_r == "all" else [skill_filter_r]

    rules_data = []
    for skill_name in rule_skills:
        for rule in forge.knowledge_base.get_rules(skill_name):
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
        st.table(rules_data)
    else:
        st.info("No rules to display")

    st.divider()

    # ── Recent errors ──────────────────────────────────────────────────────────
    st.subheader("Recent Errors")

    errors = forge.learning_engine.error_repo.get_recent(10)
    if errors:
        for err in reversed(errors):
            st.write(f"- **{err.error_type}** in {err.skill_name}: {err.task_description[:60]}...")
    else:
        st.info("No errors recorded")


if __name__ == "__main__":
    main()
