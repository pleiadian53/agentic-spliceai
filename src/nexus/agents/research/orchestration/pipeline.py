from . import agents
from ..utils import context_utils
from ..formatters import format_decision
from ..utils.progress import ProgressTracker, ProgressStage, AgentType
from typing import Optional

def generate_research_report(
    topic: str,
    model: str = "openai:gpt-4o",
    report_length: str = "standard",
    context: str = None,
    client=None,
    user_format: str = None,
    verbose: bool = True,
    progress_tracker: Optional[ProgressTracker] = None,
    enable_verification: bool = False,
) -> dict:
    """
    Orchestrates the full research workflow:
    1. Decide output format (PDF/LaTeX/Markdown)
    2. Planner creates a plan.
    3. Executor runs the plan (delegating to sub-agents).
    4. (Opt-in) Verifier anchors claims to sources; reviewer produces
       an adversarial review.

    Args:
        topic: Research topic.
        model: Model to use for agents (default gpt-4o).
        report_length: Target report length - "brief", "standard", "comprehensive", or "technical-paper"
        context: Additional context or style template guidance
        client: Optional aisuite client (for web service compatibility)
        user_format: Optional user override for format ("pdf_direct", "latex", "markdown")
        verbose: Show detailed progress (default True)
        enable_verification: If True, after the executor produces a draft,
            run the verifier (anchor claims to sources, re-search to
            verify, remove unsourced material) and the reviewer
            (FATAL/MAJOR/MINOR adversarial peer review). Default False
            — existing CLI behavior is unchanged unless callers opt in.
            The verification-suite example scripts pass True to exercise
            this path.

    Returns:
        Dictionary containing the plan, execution history, format decision,
        and (when enable_verification=True) the cited output + review +
        verification status.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"🚀 NEXUS RESEARCH AGENT")
        print(f"{'='*70}")
        print(f"📌 Topic: {topic}")
        print(f"🤖 Model: {model}")
        print(f"📏 Length: {report_length}")
        if context:
            print(f"📝 Context: {context[:100]}..." if len(context) > 100 else f"📝 Context: {context}")
        print(f"{'='*70}\n")
    
    # Progress tracking
    if progress_tracker:
        progress_tracker.update(ProgressStage.INITIALIZING, "Starting research workflow...")
    
    # Step 0: Decide output format
    format_info = format_decision.decide_output_format(
        topic=topic,
        model=model,
        user_preference=user_format
    )
    if progress_tracker:
        progress_tracker.update(ProgressStage.FORMAT_DECISION, "Deciding output format...")
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"📄 STEP 1: FORMAT DECISION")
        print(f"{'─'*70}")
        print(f"Format: {format_info['format'].upper()}")
        print(f"Reason: {format_info['reasoning']}\n")
    
    # Enhance context with smart date ranges if not specified
    enhanced_context = context_utils.enhance_context_with_dates(context)
    if enhanced_context != context and verbose:
        print(f"📅 Auto-added date range: {context_utils.get_smart_date_range()}")
    
    # 1. Plan
    if progress_tracker:
        progress_tracker.update(ProgressStage.PLANNING, "Creating research plan...")
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"📋 STEP 2: PLANNING")
        print(f"{'─'*70}")
    
    # Note: Planner typically uses reasoning model (o4-mini), but we can allow override or default.
    # The notebook used o4-mini for planning.
    planner_model = "openai:o4-mini" if model.startswith("openai") else model
    
    if verbose:
        print(f"🤖 Using planner model: {planner_model}")
        print(f"⏳ Generating research plan...\n")
    
    steps = agents.planner_agent(topic, model=planner_model, report_length=report_length, context=enhanced_context)
    
    if verbose:
        print(f"\n✅ Plan generated ({len(steps)} steps):")
        for i, step in enumerate(steps):
            print(f"  {i+1}. {step}")
        
    if not steps:
        return {"error": "Failed to generate plan"}
        
    # 2. Execute
    if progress_tracker:
        progress_tracker.update(ProgressStage.EXECUTING, f"Executing {len(steps)} research steps...")
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"⚙️  STEP 3: EXECUTION")
        print(f"{'─'*70}")
        print(f"⏳ Executing {len(steps)} research steps...")
        print(f"💡 This may take 5-10 minutes depending on complexity\n")
    
    # Get format-specific instructions for writer agent
    format_instructions = format_decision.get_writer_instructions(format_info)
    history = agents.executor_agent(
        steps, 
        model=model, 
        format_instructions=format_instructions, 
        verbose=verbose,
        progress_tracker=progress_tracker
    )
    
    # Extract final report if possible (usually the last step)
    final_output = history[-1][-1] if history else ""
    
    # Validate and fix output if needed
    from ..provenance import output_validator
    validated_output, warnings = output_validator.validate_and_fix_report(
        final_output, 
        topic, 
        auto_fix=True
    )
    
    # Log any warnings
    if warnings:
        if verbose:
            print(f"\n{'─'*70}")
            print("⚠️  OUTPUT VALIDATION")
            print(f"{'─'*70}")
            for warning in warnings:
                print(f"   {warning}")
            print()
    
    result: dict = {
        "topic": topic,
        "plan": steps,
        "history": history,
        "final_report": validated_output,
        "validation_warnings": warnings,  # Track validation issues
        "format_decision": format_info,  # Track format decision in output
    }

    # 4. (Opt-in) Verification + adversarial review
    if enable_verification:
        if verbose:
            print(f"\n{'─'*70}")
            print(f"🔍 STEP 4: VERIFICATION + REVIEW")
            print(f"{'─'*70}")
        if progress_tracker:
            progress_tracker.update(
                ProgressStage.EXECUTING, "Running verifier + reviewer..."
            )

        # Collect research context from executor history. A future
        # refactor will switch this to file-based handoffs through
        # slug-prefixed paths; for now we pass the executor history
        # as inline context.
        research_context = "\n\n---\n\n".join(
            f"### Step: {step}\n{output}"
            for step, _agent, output in history
            if _agent in {"research_agent", "writer_agent"}
        )

        # Verifier — anchors claims, re-searches with role-filtered tools
        cited_output, verification_status = agents.verifier_agent(
            draft_text=validated_output,
            research_context=research_context,
            topic=topic,
            model=model,
            return_status=True,
        )
        result["cited_output"] = cited_output
        result["verification_status"] = verification_status

        if verbose:
            print(f"   Verification: {verification_status.value}")

        # Reviewer — adversarial pass with empty tool allowlist
        review = agents.reviewer_agent(
            cited_draft_text=cited_output,
            topic=topic,
            model=model,
        )
        result["review"] = review

        if verbose:
            print(f"   Review length: {len(review)} chars")

    return result
