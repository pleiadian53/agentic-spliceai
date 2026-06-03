import json
import ast
from datetime import datetime
from typing import Optional
from aisuite import Client

from .. import tools
from nexus.llm import parsing
from nexus.llm import client as llm_client

# Shared client for aisuite calls
client = Client()

def planner_agent(topic: str, model: str = "openai:o4-mini", report_length: str = "standard", context: str = None) -> dict:
    """
    Generates a plan as a Python list of steps and decides output format.
    
    Args:
        topic: Research topic
        model: LLM model to use
        report_length: "brief" (2-3 pages), "standard" (5-10 pages), "comprehensive" (15-25 pages), "technical-paper" (25-40 pages)
        context: Additional context or style template guidance
        
    Returns:
        dict with keys:
            - 'steps': list[str] - Research plan steps
            - 'output_format': str - 'latex' or 'markdown'
            - 'reason': str - Why this format was chosen
    """
    print("==================================")
    print("🧠 Planner Agent")
    print("==================================")
    
    # Define length-specific guidance
    length_specs = {
        "brief": {
            "pages": "2-3 pages",
            "sections": "3-4 main sections",
            "depth": "High-level overview with key findings",
            "steps": "8-12 steps"
        },
        "standard": {
            "pages": "5-10 pages",
            "sections": "5-7 main sections with subsections",
            "depth": "Balanced coverage with detailed analysis",
            "steps": "12-18 steps"
        },
        "comprehensive": {
            "pages": "15-25 pages",
            "sections": "8-12 main sections with multiple subsections",
            "depth": "In-depth analysis, extensive literature review, detailed methodology",
            "steps": "20-30 steps"
        },
        "technical-paper": {
            "pages": "25-40 pages",
            "sections": "Full academic structure: Abstract, Introduction, Literature Review, Methodology, Results, Discussion, Conclusion, References",
            "depth": "Publication-quality depth with comprehensive citations, detailed analysis, and rigorous methodology",
            "steps": "30-50 steps"
        }
    }
    
    spec = length_specs.get(report_length, length_specs["standard"])
    
    # Analyze topic intent to determine focus
    topic_lower = topic.lower()
    context_lower = (context or "").lower()
    
    # Detect technical/methodological focus keywords
    technical_keywords = [
        "framework", "method", "test", "empirical", "technique", "algorithm",
        "implementation", "approach", "model", "architecture", "mechanism",
        "measurement", "evaluation", "experiment", "protocol", "procedure"
    ]
    
    # Detect ethical/societal focus keywords
    ethical_keywords = [
        "ethics", "ethical", "governance", "policy", "regulation", "safety",
        "implications", "impact", "societal", "moral", "responsibility"
    ]
    
    technical_score = sum(1 for kw in technical_keywords if kw in topic_lower or kw in context_lower)
    ethical_score = sum(1 for kw in ethical_keywords if kw in topic_lower or kw in context_lower)
    
    # Determine primary focus
    if technical_score > ethical_score:
        focus_guidance = """
🎯 PRIMARY FOCUS: Technical/Methodological Content
- Prioritize: Frameworks, methods, empirical tests, technical approaches, implementations
- Emphasize: How things work, what can be tested, new ideas and techniques
- De-emphasize: Ethical implications, governance, policy (unless explicitly requested)
- Allocate: 70-80% technical content, 20-30% broader context/implications
"""
    elif ethical_score > technical_score:
        focus_guidance = """
🎯 PRIMARY FOCUS: Ethical/Societal Implications
- Prioritize: Ethical considerations, governance frameworks, societal impact
- Emphasize: Implications, responsibilities, policy recommendations
- Balance: Technical background with ethical analysis
- Allocate: 60-70% ethical/societal content, 30-40% technical foundation
"""
    else:
        focus_guidance = """
🎯 BALANCED FOCUS: Technical and Broader Context
- Balance: Technical depth with contextual implications
- Cover: Methods, frameworks, applications, and their broader significance
- Allocate: 50-60% technical content, 40-50% context and implications
"""
    
    prompt = f"""
You are a senior research strategist orchestrating a team of specialized agents.

🎯 Your task:
Generate a valid Python list of strings, where each string is one atomic step in a multi-agent research workflow on the topic:
"{topic}"

📏 Target Report Length: {spec['pages']}
- Structure: {spec['sections']}
- Depth: {spec['depth']}
- Plan complexity: {spec['steps']}

{focus_guidance}
"""
    
    if context:
        prompt += f"""
📄 Style & Context Requirements:
{context}

Ensure the plan produces a report that matches this style and structure.
"""
    
    prompt += """
No commentary, no markdown, no surrounding backticks — only a plain Python list.

---

🧠 Available agents and what they can do:

1. **Research Agent**
   - Tools: arXiv, PubMed/Europe PMC, Tavily web search, Wikipedia
   - Capabilities: Search for papers, collect data, extract findings, compare sources, review literature, synthesize insights
   - Example verbs: "Search", "Find", "Investigate", "Collect", "Extract", "Compare", "Review", "Synthesize", "Analyze"

2. **Writer Agent**
   - Produces structured academic text: introduction, background, findings, analysis, conclusion
   - Capabilities: Draft sections, compose narratives, outline structure, expand ideas
   - Example verbs: "Draft", "Write", "Compose", "Outline", "Expand", "Develop", "Articulate"

3. **Editor Agent**
   - Improves clarity, tone, structure, correctness
   - Capabilities: Refine prose, improve clarity, restructure content, ensure accuracy
   - Example verbs: "Edit", "Refine", "Polish", "Revise", "Improve", "Enhance", "Clarify"

---

📌 Requirements for each step:
- **Atomic** → one action, one verb, one agent
- **Executable** → must clearly map to one of the above agents
- **Concrete** → specify the source or output ("Search arXiv for X", "Draft introduction section")
- **Sequenced** → broad → focused → synthesis → writing → editing
- **Complete** → include all phases: literature search, filtering, synthesis, drafting, editing, final report

📌 Step formatting:
- Each element is a **string** describing one action
- Each string should start with an action verb (see examples above)
- No compound steps with connectors ("and", "then", "followed by")
- Avoid infrastructure tasks (code, repos, datasets)
- Avoid meta instructions ("think about", "consider", "reflect on")

---

📌 Recommended structure:
1. Broad literature search (multiple sources)
2. Narrowed academic search (specific aspects)
3. Complementary web/encyclopedic search (context, definitions)
4. Extraction/summarization steps (key findings)
5. Synthesis/insights (cross-reference, compare)
6. Writing steps (multiple sections: intro, findings, analysis, conclusion)
7. Editing/refinement (clarity, structure, accuracy)
8. Final report generation

---

📌 Requirements for each step:
- **Atomic** → one action, one verb, one agent
- **Executable** → must clearly map to one of the above agents
- **Concrete** → specify the source or output ("Search arXiv for X", "Draft introduction section")
- **Sequenced** → broad → focused → synthesis → writing → editing
- **Complete** → include all phases: literature search, filtering, synthesis, drafting, editing, final report

Now generate the plan for: "{topic}"
"""
    
    if context:
        prompt += f"""
📄 Style & Context Requirements:
{context}

Ensure the plan produces a report that matches this style and structure.
"""
    
    messages = [{"role": "user", "content": prompt}]
    
    # Use unified client (supports Responses API)
    content = llm_client.call_llm_text(client, model, messages, temperature=1.0)
    
    try:
        # Clean markdown blocks if present
        content = parsing.clean_json_block(content)
        
        # Parse list safely
        steps = ast.literal_eval(content)
        if not isinstance(steps, list):
            raise ValueError("Output is not a list")
            
        return steps
    except Exception as e:
        print(f"❌ Planner Error: {e}")
        return []

def research_agent(task: str, model: str = "openai:gpt-4o", return_messages: bool = False) -> str | tuple[str, list]:
    """
    Executes a research task using tools.
    Delegates to llm_client.call_llm_with_tools for API compatibility.
    """
    print("==================================")
    print("🔍 Research Agent")
    print("==================================")

    prompt = f"""
You are an expert research analyst with access to cutting-edge academic and web search tools.

🔬 Your capabilities:
- **arxiv_search_tool**: Access to 2M+ papers in physics, CS, math, and related fields
- **europe_pmc_search_tool**: Comprehensive biomedical literature (PubMed, bioRxiv, medRxiv, preprints)
- **tavily_search_tool**: Real-time web search for current events, news, and general information
- **wikipedia_search_tool**: Encyclopedic knowledge for background context and definitions

🎯 Your mission:
{task}

📋 Best practices:
- Use multiple tools to cross-reference and validate findings
- Prioritize recent publications (especially for fast-moving fields)
- Cite specific papers with titles, authors, and publication years
- Synthesize findings into coherent insights, not just lists
- Be critical: note limitations, controversies, or gaps in the literature

📅 Today's date: {datetime.now().strftime('%Y-%m-%d')}

🚀 Begin your research now. Use tools strategically and provide a comprehensive, well-sourced response.
"""
    messages = [{"role": "user", "content": prompt.strip()}]
    
    # Delegate to unified client
    content = llm_client.call_llm_with_tools(
        client=client,
        model=model,
        messages=messages,
        aisuite_tools=tools.aisuite_tools,
        responses_tool_defs=tools.responses_tool_defs,
        tool_mapping=tools.tool_mapping
    )
    
    print("✅ Output:\n", content)
    return (content, messages) if return_messages else content

def writer_agent(task: str, model: str = "openai:gpt-4o", format_instructions: str = "") -> str:
    """
    Executes writing tasks with optional format-specific instructions.
    
    Args:
        task: Writing task description
        model: LLM model to use
        format_instructions: Additional instructions for specific output format (LaTeX, Markdown, etc.)
    """
    print("==================================")
    print("✍️ Writer Agent")
    print("==================================")
    
    system_prompt = """You are an award-winning technical writer with expertise in academic publications, grant proposals, whitepapers, and research reports.

Your writing is characterized by:
- Clear, precise language that balances accessibility with technical rigor
- Logical flow with smooth transitions between ideas
- Evidence-based arguments supported by citations
- Well-structured sections with informative headings
- Engaging introductions that establish context and significance
- Insightful conclusions that synthesize findings and suggest future directions
- Compelling narratives that make science interesting and attractive to readers
- Concrete examples and analogies that illuminate abstract concepts
- Real-world applications that demonstrate practical relevance

Your writing philosophy:
- Make complex ideas accessible through clear explanations and relatable examples
- Use storytelling techniques to maintain reader engagement
- Provide intuitive analogies when introducing technical concepts
- Include illustrative examples that readers can visualize
- Balance depth with readability—never sacrifice clarity for jargon
- Motivate the "why" before diving into the "how"
- Connect abstract theory to tangible applications

Your output formats include:
- Scientific publications (Nature, Science, IEEE style)
- Grant proposals (NSF, NIH, industry RFPs)
- Technical whitepapers and reports
- Research summaries and literature reviews

Write with authority, clarity, and intellectual depth. Make science come alive—avoid dry, lifeless prose. Your goal is to inform, engage, and inspire."""
    
    # Add format-specific instructions if provided
    if format_instructions:
        system_prompt += "\n\n" + format_instructions
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    content = llm_client.call_llm_text(client, model, messages, temperature=1.0)
    return content

def editor_agent(task: str, model: str = "openai:gpt-4o", format_instructions: str = "") -> str:
    """
    Executes editorial tasks.
    
    Args:
        task: Editorial task to perform
        model: LLM model to use
        format_instructions: Format-specific instructions (LaTeX, Markdown, etc.)
    """
    print("==================================")
    print("🧠 Editor Agent")
    print("==================================")
    
    system_prompt = """You are a senior editor with decades of experience in academic peer review, technical editing, and content refinement.

Your editorial expertise covers:
- Structural coherence: Ensure logical flow and clear argumentation
- Clarity and precision: Eliminate ambiguity, jargon, and verbosity
- Evidence quality: Verify claims are well-supported and citations are appropriate
- Technical accuracy: Catch errors, inconsistencies, or unsupported assertions
- Readability: Balance technical depth with accessibility for the target audience
- Style consistency: Maintain professional tone and formatting standards

Your editorial approach:
1. Review the draft for strengths and weaknesses
2. Make specific improvements to structure, clarity, evidence, and style
3. Rewrite sections that need substantial revision
4. Return the COMPLETE EDITED REPORT with all improvements applied
5. Ensure the final product meets publication standards

CRITICAL OUTPUT REQUIREMENT:
- Your output MUST be the full, final, publication-ready document
- Do NOT return editorial commentary, feedback, or instructions
- Do NOT include meta-discussion about the editing process
- Return ONLY the polished, complete report itself

Be constructive but rigorous. Your goal is to elevate good writing to excellence and deliver the final product."""
    
    # Add format-specific instructions if provided
    if format_instructions:
        system_prompt += "\n\n" + format_instructions
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    content = llm_client.call_llm_text(client, model, messages, temperature=0.7)
    return content

# Agent registry for executor
agent_registry = {
    "research_agent": research_agent,
    "editor_agent": editor_agent,
    "writer_agent": writer_agent,
}

def executor_agent(
    plan_steps: list[str], 
    model: str = "openai:gpt-4o", 
    format_instructions: str = "", 
    verbose: bool = True,
    progress_tracker: Optional['ProgressTracker'] = None  # This value can be of type ProgressTracker, or it can be None
):
    """
    Routes each task to the correct sub-agent.
    
    Args:
        plan_steps: List of research plan steps
        model: LLM model to use
        format_instructions: Format-specific instructions for writer agent
        verbose: Show detailed progress (default True)
        progress_tracker: Optional progress tracker for real-time updates
    """
    from ..utils.progress import AgentType
    from typing import Optional
    
    history = []

    for i, step in enumerate(plan_steps):
        if verbose:
            print(f"\n{'┌'+'─'*68+'┐'}")
            print(f"│ 📍 Step {i+1}/{len(plan_steps)}: {step[:60]+'...' if len(step) > 60 else step:<60} │")
            print(f"{'└'+'─'*68+'┘'}")
        agent_decision_prompt = f"""
You are an execution manager for a multi-agent research team.

Given the following instruction, identify which agent should perform it and extract the clean task.

Return only a valid JSON object with two keys:
- "agent": one of ["research_agent", "editor_agent", "writer_agent"]
- "task": a string with the instruction that the agent should follow

Only respond with a valid JSON object. Do not include explanations or markdown formatting.

Instruction: "{step}"
"""
        messages = [{"role": "user", "content": agent_decision_prompt}]
        content = llm_client.call_llm_text(client, model, messages, temperature=0.0)

        try:
            cleaned_json = parsing.clean_json_block(content)
            agent_info = json.loads(cleaned_json)

            agent_name = agent_info["agent"]
            task = agent_info["task"]

            # Build context
            context = "\n".join([
                f"Step {j+1} executed by {a}:\n{r}" 
                for j, (s, a, r) in enumerate(history)
            ])
            enriched_task = f"""You are {agent_name}.

Here is the context of what has been done so far:
{context}

Your next task is:
{task}
"""

            # Map agent names to types and info
            agent_type_map = {
                "research_agent": AgentType.RESEARCH,
                "writer_agent": AgentType.WRITER,
                "editor_agent": AgentType.EDITOR
            }
            
            agent_info_map = {
                "research_agent": ("🔍", "Research Agent", "Gathering knowledge"),
                "writer_agent": ("✍️", "Writer Agent", "Generating content"),
                "editor_agent": ("📝", "Editor Agent", "Refining quality")
            }
            
            # Report progress
            if progress_tracker:
                progress_tracker.update_step(
                    step_number=i+1,
                    total_steps=len(plan_steps),
                    agent=agent_type_map.get(agent_name, AgentType.RESEARCH),
                    message=f"{task[:100]}"
                )
            
            if verbose:
                emoji, name, action = agent_info_map.get(agent_name, ("🤖", agent_name, "Processing"))
                print(f"{emoji} {name}: {action}")
                print(f"   Task: {task[:80]+'...' if len(task) > 80 else task}")

            if agent_name in agent_registry:
                # Pass format_instructions to writer_agent and editor_agent
                if agent_name in ["writer_agent", "editor_agent"] and format_instructions:
                    output = agent_registry[agent_name](enriched_task, model=model, format_instructions=format_instructions)
                elif agent_name == "research_agent":
                    output = agent_registry[agent_name](enriched_task)
                else:
                    output = agent_registry[agent_name](enriched_task, model=model)
                history.append((step, agent_name, output))
                
                if verbose:
                    # Show abbreviated output
                    output_preview = output[:200] + "..." if len(output) > 200 else output
                    print(f"   ✅ Complete ({len(output)} chars)")
            else:
                output = f"⚠️ Unknown agent: {agent_name}"
                history.append((step, agent_name, output))
                if verbose:
                    print(f"   ❌ Error: Unknown agent")
            
        except Exception as e:
            print(f"❌ Execution Error at step {i}: {e}")
            history.append((step, "error", str(e)))

    return history


# ---------------------------------------------------------------------------
# Verifier + reviewer — role-aware agents with mechanical tool restriction
# ---------------------------------------------------------------------------
#
# These two functions load their role contract from `roles/<name>.md`, build
# role-filtered tool schemas via `tools.role_filter`, and invoke the LLM with
# ONLY the tools the role's allowlist permits. The writer/reviewer have empty
# allowlists, so they receive zero tool schemas — the LLM physically cannot
# emit a tool call for any tool. A role cannot solve a problem with the same
# level of thinking that created it; here that's enforced by the schema
# filter, not by prompt discipline.
#
# Smoke-tested in `examples/nexus/04_role_tool_restrictions.py` (no live LLM
# call needed for the structural assertion).


def verifier_agent(
    draft_text: str,
    research_context: str,
    topic: str,
    model: str = "openai:gpt-4o",
    return_status: bool = False,
):
    """Anchor every factual claim in the draft to a source from the research files.

    Loads the verifier role contract, applies role-restricted tools
    (search-only; no authoring), invokes the LLM, returns the cited
    draft. Optionally returns a `(text, VerificationStatus)` tuple.

    Args:
        draft_text: the unsourced writer draft.
        research_context: research findings (concatenated research files
            or executor history) the verifier can anchor claims to.
        topic: the original research topic.
        model: LLM to use.
        return_status: if True, returns (cited_text, VerificationStatus).
    """
    from ..roles import load_role
    from ..tools import role_filter
    from ..tools.tools import (
        responses_tool_defs,
        tool_mapping as global_tool_mapping,
        aisuite_tools as global_aisuite_tools,
    )
    from ..provenance import VerificationStatus
    from nexus.llm.tool_loop import call_llm_with_tools

    role = load_role("verifier")
    role_schemas = role_filter.build_tool_schemas_for_role(role, responses_tool_defs)
    role_map = role_filter.filter_tool_mapping_for_role(role, global_tool_mapping)
    role_aisuite = role_filter.filter_aisuite_tools_for_role(role, global_aisuite_tools)

    user_msg = (
        f"Topic: {topic}\n\n"
        f"## Draft to verify\n\n{draft_text}\n\n"
        f"## Research context\n\n{research_context}\n\n"
        f"Apply your verification contract:\n"
        f"- Anchor every factual claim to a research source with [N] citations.\n"
        f"- Verify URLs by re-searching when possible (you have search tools).\n"
        f"- Remove unsourced claims or convert to TODOs.\n"
        f"- Build the final Sources section.\n"
        f"- If a claim cannot be verified, mark it `BLOCKED:` in-line and "
        f"explain why."
    )

    messages = [
        {"role": "system", "content": role.system_prompt},
        {"role": "user", "content": user_msg},
    ]

    result = call_llm_with_tools(
        client=client,
        model=model,
        messages=messages,
        aisuite_tools=role_aisuite,
        responses_tool_defs=role_schemas,
        tool_mapping=role_map,
    )

    if return_status:
        text = result if isinstance(result, str) else str(result)
        status = (
            VerificationStatus.BLOCKED
            if "BLOCKED" in text
            else VerificationStatus.PASS
        )
        return text, status
    return result


def reviewer_agent(
    cited_draft_text: str,
    topic: str,
    model: str = "openai:gpt-4o",
) -> str:
    """Adversarial peer review of the cited draft. Read-only role.

    Loads the reviewer role contract (empty tool allowlist), invokes
    the LLM with NO tool schemas in the request payload (so the model
    cannot emit tool calls), returns a FATAL/MAJOR/MINOR review.
    """
    from ..roles import load_role
    from nexus.llm.client import call_llm_text

    role = load_role("reviewer")
    user_msg = (
        f"Topic: {topic}\n\n"
        f"## Cited draft to review\n\n{cited_draft_text}\n\n"
        f"Apply your reviewer contract:\n"
        f"- Tag findings as FATAL / MAJOR / MINOR.\n"
        f"- Keep looking after the first issue — read the whole draft.\n"
        f"- Focus on logical-level integrity, not citation-level "
        f"(the verifier already did that).\n"
        f"- Produce the structured review document (Summary / Strengths / "
        f"Weaknesses / Questions / Verdict / Revision Plan)."
    )
    messages = [
        {"role": "system", "content": role.system_prompt},
        {"role": "user", "content": user_msg},
    ]
    return call_llm_text(client, model, messages, temperature=0.3)


__all__ = [
    "planner_agent",
    "research_agent",
    "writer_agent",
    "editor_agent",
    "executor_agent",
    "verifier_agent",
    "reviewer_agent",
]
