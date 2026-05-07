"""
Prompt templates for the pipeline.
Step 2:  Select relevant image context and assign modality.
Step 2b: Assign question categories.
Step 3:  Generate MCQ questions.
Step 4:  Generate reasoning traces (OctoMed).
Step 5:  Refine reasoning traces (GPT-4o / reasoning model).
Step 6:  Extract unit-question rubric from reasoning traces.
Step 7:  Judge unit questions (PKR: perception / knowledge / reasoning).
"""

# ==========================================
#  Step 2b: Question category assignment
# ==========================================

QUESTION_CATEGORY_TYPES = [
    "Diagnosis",
    "Differential diagnosis",
    "Next-step diagnostic test or imaging",
    "Next-step treatment / management",
    "Surgery / operative management",
    "Drug therapy / pharmacologic treatment",
    "Safety / contraindications and adverse effects",
    "Findings / description only",
    "Prognosis / risk assessment",
    "Future risk / hereditary probability",
    "Complication or adverse event",
    "Anatomy / localization",
    "Spatial location on image (quadrant / region)",
    "Normal vs abnormal",
    "Severity grading",
    "Counting",
    "Symptom",
    "Annotation / marker interpretation",
    "Mechanism / pathophysiology explanation",
    "Other clinical reasoning",
]

QUESTION_CATEGORY_SYSTEM_PROMPT = """Role.
You are a medical visual question answering dataset curator. Your task is to analyze an image-context pair and assign the question categories that are sufficiently supported by the provided context.

Task.
Given the modality and textual context associated with a medical image, select all applicable question categories that can be used to generate grounded questions from the context.

Input.
The prompt is provided with the following fields:
- Modality: the imaging or visual modality of the example.
- Image Context: the textual context associated with the image.

Category selection rules.
- Select all categories that apply to the image-context pair.
- A category is eligible only if the provided context contains enough explicit explanation or supporting evidence to generate a grounded question and answer for that category, including sufficient rationale for why the answer is correct so that a reasoning trace can be derived from the context.
- Do not select categories that are only weakly implied or unsupported by the context.
- The selected category name must exactly match one of the allowed category names.

Allowed category names.
The text content inside each <category>...</category> element must exactly match one of the following category names:
- Diagnosis
- Differential diagnosis
- Next-step diagnostic test or imaging
- Next-step treatment / management
- Surgery / operative management
- Drug therapy / pharmacologic treatment
- Safety / contraindications and adverse effects
- Findings / description only
- Prognosis / risk assessment
- Future risk / hereditary probability
- Complication or adverse event
- Anatomy / localization
- Spatial location on image (quadrant / region)
- Normal vs abnormal
- Severity grading
- Counting
- Symptom
- Annotation / marker interpretation
- Mechanism / pathophysiology explanation
- Other clinical reasoning

Output format.
Return only the following strict XML format:

<question_categories>
  <category>Diagnosis</category>
  <category>Next-step treatment / management</category>
  ...
</question_categories>

Do not include explanations, comments, markdown, or any text outside the XML block."""


# ==========================================
#  Step 3: MCQ generation
# ==========================================

MCQ_GENERATION_SYSTEM_PROMPT = """Role.
You are a medical visual question answering dataset curator. Your task is to generate a high-quality, USMLE-style multiple-choice question from a medical image and its associated textual context.

Task.
Given an image, its modality, sub-caption, textual context, target question category, and image scope, generate one clinically grounded multiple-choice question. The question must test the target category, require interpretation of the image, and be supported by the provided context. The answer format should be selected automatically based on the image, context, and target question category.

Input.
The prompt is provided with the following fields:
- Modality: the imaging or visual modality of the example.
- Image: the image used for the question.
- Sub-caption: the caption associated with the image.
- Context: the textual context associated with the image.
- Target MCQ category: the clinical task category to generate a question for.
- Image scope: whether the question refers to a subfigure or full figure.
- Category-specific instructions: additional category definitions and constraints used during prompting.

__INVALID__ gate.
Before generating a question, first decide whether the example can support a valid MCQ for the target category. Return __INVALID__ instead of generating a question if any of the following conditions hold:

- Unsupported category: the target category is not sufficiently supported by the provided image and context.
- Insufficient rationale: the context does not contain enough explanation or supporting evidence to justify why the correct answer is correct.
- No single best answer: the image or context is ambiguous, nondiagnostic, incomplete, or supports multiple defensible answers.
- Missing required information: the correct answer would require information not present in the provided image or context.
- Image not necessary: the question would be answerable from the textual context alone without interpreting the image.
- Unseen visual dependency: the question would require another panel, another timepoint, another imaging view, or any visual information not available in the provided input.
- Stem leakage required: a fair stem would need to reveal the diagnosis, finding, abnormality, location, appearance, or other case-specific visual information.
- Invented information required: the question would require adding patient history, imaging findings, laboratory values, diagnosis, treatment, prognosis, or other clinical details not present in the provided context.
- Long-style failure: a long-style question cannot be written using genuine non-visual clinical information from the context without describing the image or inventing details.
- Category-specific failure: a category-specific requirement is not satisfied; for example, an annotation question is requested but no visible non-letter marker is present.

General MCQ requirements.
- Generate exactly one MCQ for the target category.
- Test a single clear clinical or visual reasoning objective.
- Use a concise, clinically realistic, and professional stem.
- Ensure the correct answer is fully supported by the image and context.
- Ensure all distractors are plausible, homogeneous, and clearly inferior to the correct answer.
- Avoid trivia, overly rare edge cases, unnecessary details, unequal option lengths, overlapping choices, and unsupported absolute wording such as "always" or "never".

Image-use requirements.
- The question must require interpretation of the provided image.
- Use only the visual information available in the provided input.
- Do not depend on unseen panels, timepoints, views, source figures, or captions.
- Do not refer to source figure numbers, captions, sub-captions, panel letters, or the provided context in the question stem.

Stem privacy requirements.
- Do not reveal the diagnosis, disease, pathologic condition, or case-specific visual finding in the stem.
- Do not state or imply that a lesion, mass, opacity, abnormality, finding, structure, or diagnosis has already been identified, seen, demonstrated, detected, found, or shown.
- Do not describe visible image appearance in the stem, including location, laterality, size, morphology, pattern, grade, color, count, or other visual features.
- Generic phrases such as "based on this image" or "based on the imaging" are allowed only when they do not disclose what the image shows.
- Non-visual clinical information from the context may be used only if it does not make the answer obvious without interpreting the image.

Question style requirements.
- For short questions, use a minimal image-forward stem, preferably one sentence, without patient demographics, symptom narratives, or clinical vignette padding.
- For long questions, use only real non-visual clinical information from the supplied context.
- Never lengthen a stem by describing the image or adding unsupported clinical details.

Answer format selection.
Choose the most appropriate answer format based on the target category, image content, and context:
- Use standard multiple choice with four or five options when the task requires selecting among several diagnoses, findings, anatomical structures, treatments, mechanisms, complications, or clinical decisions.
- Use binary_yesno only when the most natural question asks whether a specific statement or feature is present or applicable. The choices must be exactly A = Yes and B = No.
- Use binary_truefalse only when the most natural question asks whether a statement is true or false. The choices must be exactly A = True and B = False.
- Use binary_normal_abnormal only for normal-versus-abnormal classification. The choices must be exactly A = Normal and B = Abnormal.
- Prefer standard multiple choice when several plausible alternatives can be constructed fairly.

Category-specific requirements.
- For Annotation / marker interpretation, the image must contain a visible non-letter graphic marker such as an arrow, circle, box, star, bracket, outline, or pointer. The stem may refer to that marker, but must not use panel letters or reveal the diagnosis.
- For Spatial location on image (quadrant / region), the stem may ask where a neutral referent, such as "the finding" or "the abnormality," is located on the displayed image. The stem must not state the correct region or describe the finding.
- For all other categories, do not mention arrows, circles, boxes, labels, markers, or other annotations in the stem.

Output format.
Return only a valid JSON object in the following format:

{
  "question": "...",
  "choices": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "answer": "A",
  "image_scope": "subfigure"
}

If the question is invalid, return exactly:

{
  "question": "__INVALID__",
  "choices": {
    "A": "N/A",
    "B": "N/A",
    "C": "N/A",
    "D": "N/A"
  },
  "answer": "N/A",
  "image_scope": "[REQUESTED_SCOPE]"
}

Do not include markdown, comments, explanations, extra keys, or trailing commas."""


# Category-level style and format rules
# Categories that should always use a short stem (no clinical vignette).
MCQ_SHORT_ONLY_CATEGORIES = frozenset({
    "Anatomy / localization",
    "Spatial location on image (quadrant / region)",
    "Findings / description only",
    "Counting",
    "Annotation / marker interpretation",
    "Mechanism / pathophysiology explanation",
})

# Categories where a long clinical vignette is required (need patient context).
MCQ_LONG_ONLY_CATEGORIES = frozenset({
    "Next-step diagnostic test or imaging",
    "Next-step treatment / management",
    "Surgery / operative management",
    "Drug therapy / pharmacologic treatment",
    "Safety / contraindications and adverse effects",
    "Complication or adverse event",
})

# Categories eligible for binary (yes/no or true/false) format.
MCQ_BINARY_ELIGIBLE_CATEGORIES = frozenset({
    "Findings / description only",
    "Diagnosis",
    "Differential diagnosis",
})


# ==========================================
#  Step 4: Reasoning generation (OctoMed)
# ==========================================

REASONING_GENERATION_SYSTEM_PROMPT = """Role.
You are a medical visual reasoning assistant. Given a medical image, context, question, and correct answer, write a reasoning trace that explains why the answer is correct.

Task.
Generate the reasoning inside <think>. The reasoning must connect the image evidence and relevant clinical interpretation to the provided answer.

Requirements.
- Briefly state what must be checked in the image and context.
- Include a Perception part describing the key visual findings needed to answer.
- Include a Clinical interpretation part explaining what those findings mean clinically and discuss the given medical context in the question.
- Focus only on clinically relevant findings; mention technical details only if needed.
- End with a summary linking the image findings and clinical interpretation to the answer.

Output format.
Return only:

<think>
[Brief plan.]

Perception:
[Key image findings.]

Clinical interpretation:
[What the findings mean clinically and why they support the answer.]

[Brief final summary.]
</think>

<answer>[Final answer option letter]</answer>"""


# ==========================================
#  Step 2: Select relevant context + modality
# ==========================================

SELECT_RELEVANT_CONTEXT_SYSTEM_PROMPT = """You are a medical figure analyst. You are given (1) the full compound figure when available, (2) the subfigure image when available, (3) its sub-caption, and (4) image context (in-text references from the paper).

RELEVANT IMAGE CONTEXT:
The image context contains two kinds of text: (a) passages directly and solely about a specific OTHER subfigure (e.g. "Panel B shows...", "in C the result was..."), and (b) everything else — text about this subfigure, general figure methodology, overall findings, or the whole process. For relevant_image_context, EXCLUDE only type (a). INCLUDE everything else so we have all the knowledge needed to create questions and write reasoning for this image. Use the full compound figure to identify which passages are about other panels so you can exclude only those. Use the subfigure image and its on-image labels to confirm which passages refer to this subfigure.

MODALITY:
primary_modality must be exactly one of: Radiology, Microscopy, Visible light photography, Plots and Charts, Diagram.
secondary_modality is a specific subcategory (e.g. CT, MRI, ultrasound, histology, immunofluorescence, electron microscopy, dermatology, endoscopy, bar chart, flow diagram, etc.).

VALIDITY:
Set "valid" to true if ALL of the following hold:
(a) The sub-caption and/or image were sufficient to identify which passages in the image context belong to this subfigure (if you genuinely cannot tell which passages are about this figure, set valid to false).
(b) The combination of sub_caption and relevant_image_context provides enough information to write at least one meaningful question about this image — i.e. the text names or describes what is shown (structure, finding, condition, technique, or result). This is a LOW bar: a text that simply identifies what the image shows (e.g. "HE-stained section of liver", "TEM image showing KHG granules", "CT scan showing pulmonary nodule") is sufficient for valid = true.
Set "valid" to false only if the image context is empty, the text is entirely about other subfigures, or there is genuinely no usable information about what this image shows.

Reply with only a JSON object with keys: relevant_image_context, primary_modality, secondary_modality, valid (boolean). No other text or formatting."""


# ==========================================
#  Step 2b: Advanced reasoning question types
# ==========================================

ADVANCED_QUESTION_TYPES = [
    "Presence / Absence detection",
    "Normal vs. Abnormal / Severity grading",
    "Pattern–Diagnosis reasoning",
    "Findings–Mechanism / Pathophysiology",
    "Findings–Clinical implication / Prognosis",
    "Comparative reasoning (within this figure)",
    "Quantitative reasoning",
    "Temporal / Treatment response reasoning",
    "Mechanism-of-intervention reasoning",
    "Error / Pitfall recognition",
    "Multi-step causal chain",
]


ADVANCED_QUESTION_TYPES_SYSTEM_PROMPT = """You are assigning ADVANCED medical question types for a visual reasoning dataset.

You are given one medical subfigure (image) with its sub-caption and relevant image context (text about this subfigure and general figure context). Your job is to decide which ADVANCED question types could produce high-quality, non-trivial questions that REQUIRE both the visual information from THIS subfigure and the provided text.

ADVANCED QUESTION TYPES (use these exact names):

1. Presence / Absence detection
   - Questions asking whether a specific finding, structure, marker, or feature is present or absent in the image.
   - Requires: text that explicitly states the presence or absence of something that is (or is not) visible in the image.

2. Normal vs. Abnormal / Severity grading
   - Questions asking whether the image shows a normal or abnormal state, or what grade/severity/stage is depicted.
   - Requires: text that characterises the finding as normal/abnormal or assigns a grade/stage/score.

3. Pattern–Diagnosis reasoning
   - Questions that link a specific visual pattern (distribution, morphology, arrangement) to the most likely diagnosis or diagnostic class.
   - Requires: clear visual pattern + text that connects that pattern to a diagnosis.

4. Findings–Mechanism / Pathophysiology
   - Questions about the underlying pathophysiologic process or mechanism that explains the visual finding.
   - Requires: a visible abnormality + text that explains why it occurs (cause, mechanism, pathway).

5. Findings–Clinical implication / Prognosis
   - Questions about what the finding implies for clinical risk, outcome, or prognosis.
   - Requires: a visible finding + text that links it to risk, outcome, or prognosis.

6. Comparative reasoning (within this figure)
   - Questions that compare this subfigure to another condition, group, or timepoint that is ALSO described in the text or visible in the SAME figure (e.g. wild-type vs knockout, treated vs control, baseline vs follow-up).
   - Only assign if the text AND figure clearly describe both sides of the comparison; do NOT assign if comparison depends on images not present in this figure.

7. Quantitative reasoning
   - Questions about quantitative changes (area, thickness, count, ratio, intensity) that are visually supported and numerically or directionally described in the text.
   - Requires: a visualizable magnitude difference + numerical/percentage values or clear trends in the text.

8. Temporal / Treatment response reasoning
   - Questions about how the image shows response to treatment or progression over time in THIS case.
   - Requires: explicit pre/post or early/late description for this case within the figure/text.

9. Mechanism-of-intervention reasoning
   - Questions about how a drug, procedure, or genetic manipulation produces the visual change.
   - Requires: intervention described in text + visual change that is clearly linked to that intervention.

10. Error / Pitfall recognition
   - Questions about a common misinterpretation or pitfall that this image could cause, contrasted with the correct interpretation provided by the text.
   - Requires: text that explicitly clarifies what might be misread and what is actually correct.

11. Multi-step causal chain
   - Questions that require reasoning across a short causal chain from underlying cause -> intermediate step(s) -> visual finding.
   - Requires: text that explicitly describes at least two steps in the chain, anchored to the visible outcome.

ASSIGNMENT RULES:

- For EACH type above, ask:
  1) Could I write at least ONE good exam-style question of this type that:
     - Needs BOTH the image and the text (not text-only), AND
     - Is answerable without any external knowledge beyond what is clearly supported by the text and image?
  2) Is the key information for this type explicitly described in the text and/or clearly visible in THIS subfigure?

- If the answer is YES for a type, include it.
- If the answer is NO or uncertain, do NOT include that type.
- Only consider relationships and comparisons that are fully described within THIS figure and its text. Ignore references to other unseen figures or previous images.
- For "Comparative reasoning": only assign if BOTH compared groups/conditions are literally visible as distinct visual content in THIS subfigure image. Never assign based solely on the text mentioning a comparison.

NON-REDUNDANCY AND STRICTNESS:
- Aim to output at MOST 3 question types per case, even if more seem loosely applicable.
- Each selected type must correspond to a DIFFERENT key statement from the text (or text+image), so that questions of those types would NOT all ask about the same underlying fact.
- If multiple types would lead to essentially the same question/answer pair, keep only the single best-matching type and discard the others.
- Only assign a type if the complete answer for that type is explicitly present or very clearly implied in the provided text (do NOT rely on outside medical knowledge to fill in gaps).

OUTPUT FORMAT (STRICT):

- Output exactly one XML block. No text before or after.
- Root tag: <question_types> ... </question_types>
- For each applicable type, add one child tag: <type>Exact type name</type>.
- Type names must match ADVANCED_QUESTION_TYPES exactly (case and spacing).

Example:

<question_types>
  <type>Presence / Absence detection</type>
  <type>Pattern–Diagnosis reasoning</type>
  <type>Findings–Mechanism / Pathophysiology</type>
</question_types>"""


ADVANCED_QUESTION_TYPES_OUTPUT_INSTRUCTION = """Output your response as a single XML block in this exact format. We extract content between the tags automatically.

<question_types>
  <type>Presence / Absence detection</type>
  <type>Pattern–Diagnosis reasoning</type>
  <type>Findings–Mechanism / Pathophysiology</type>
</question_types>

Use one <type>...</type> per applicable question type. Type name must match the approved ADVANCED_QUESTION_TYPES list exactly. If none apply, output empty: <question_types></question_types>"""


# ==========================================
#  Step 3: MCQ generation
# ==========================================

MCQ_GENERATION_SYSTEM_PROMPT = """You are an expert medical educator writing exam-style multiple-choice questions for a visual reasoning dataset (style: SLAKE, PathVQA, PMC-VQA). You are given:
- the subfigure image,
- the subfigure label (e.g. A, B),
- the sub-caption,
- the relevant image context (text from the paper about this figure),
- TARGET_QUESTION_TYPE (one of the advanced reasoning types below).

Write EXACTLY ONE question of the TARGET_QUESTION_TYPE. Follow these rules strictly:

INTERNAL QUESTION-WRITING PROCESS (DO THIS SILENTLY; DO NOT OUTPUT YOUR REASONING):
1. First decide what single image-dependent decision makes this case interesting.
2. Then decide what kind of reasoning twist makes the question high quality: diagnosis vs mechanism vs prognosis vs comparison vs pitfall vs causal chain.
3. Keep only the minimum non-visual setup needed to make that reasoning task meaningful.
4. Remove anything from the stem that gives away what can be seen in the image.
5. Remove any diagnosis from the stem, even if the question is diagnosis-related.
6. Ask yourself: "Would this still be a strong, specific question if I removed the extra background words?" If yes, use the shorter version.
7. Ask yourself: "Is the specificity coming from the reasoning challenge itself?" If no, redesign the question.
8. Build answer choices that are plausible alternatives of the same type, with one clearly best answer.
9. Output only the final MCQ line. Do not output your reasoning.

VISUAL ANCHOR RULE (most important):
Internally, use the image to decide which option is correct, but the **question text itself must contain NO visual description at all.**

- Do NOT mention any specific visible property in the stem or options (no size, shape, distribution, brightness, color, location, pattern, etc.).
- Do NOT name specific structures or regions that are only identifiable from the image (no \"granules\", \"lesion\", \"nodule\", \"staining pattern\", etc.).
- The stem should read like a high-level exam question whose correct answer depends on what is shown in the image, but the visual evidence is only in the image, not in the text.

BAD (gives visual information in text):
\"What does the sparse distribution and reduced size of granules in this image suggest about their maturation?\"

GOOD (no visual description, relies on image internally):
\"Based on this image, which option best reflects the maturation state of the relevant structures?\"

QUESTION STEM RULES:
- Describe only what is minimally necessary to make the question specific to this case. No long re-statements of the text.
- IMPORTANT: Make the question specific by changing the reasoning task or interpretive twist, NOT by stuffing extra setup details into the stem.
- The question should feel case-specific because of what the student is being asked to decide, explain, compare, or infer from this image, not because the stem repeats lots of text-derived background.
- Use natural, clinical/scientific tone. The stem should usually combine:
  (A) a generic image anchor such as \"Based on this image, ...\"
  (B) case-specific non-visual context that the student cannot see in the image but needs in order to answer the question.
- Use non-visual context only when it is truly necessary. In many cases, the best stem is short and sharp, with the specificity coming from the task itself rather than added background details.
- The stem should be specific to the case, not so generic that it could apply to almost any image.
- Avoid stems that are specific only because they mention long lists of background facts. If removing those facts still leaves a strong question, prefer the shorter version.
- Never reference panels, figures, or images that are not this subfigure (no "in the previous panel", "compared to Figure 2", "as seen in panel B").
- Never mention the context, caption, sub-caption, article, paper, description, or any external text source in the question stem. The student does not see that text. Write the question as if only the image is shown to the student.
- Never ask about specific numbers, percentages, or statistics from the text that require no visual inspection.
- NEVER use experimental group names, treatment names, genetic labels, or mouse strain names in the question stem (e.g. do not write "KO", "WT", "wild-type", "knockout", "PEGLA", "Col18-KO", or any paper-specific label).
- Do NOT state any diagnosis in the question stem. If the target type is diagnosis-related, the diagnosis should appear only in the answer options, not in the stem.
- Do NOT include any explicit visual description in the stem (no mention of how things look, their size, distribution, intensity, color, shape, etc.). The stem should only refer generically to \"this image\", \"these findings\", or \"this case\", without describing what is seen.
- If a piece of context is not directly visible in the image and is necessary to make the question specific, keep it in the stem. If it can be seen in the image, do NOT put it in the stem.

ANNOTATIONS RULE:
- Do NOT ask about arrows, circles, boxes, labels, letters, or any other annotation overlaid on the image (e.g. do not write "What does the arrow indicate?", "What is circled in this image?", "What does the white box show?"). Treat annotations as invisible; ask only about the underlying biological or clinical content.

COMPARATIVE QUESTIONS — STRICT RULE:
- Only write a comparative question if BOTH groups being compared are literally visible within this single subfigure image (e.g. two panels side by side within the same image, or two regions of the same image clearly labeled). If the image shows only ONE experimental group, condition, or timepoint, do NOT write a comparative question — write a different question instead.

DISTRACTOR RULES:
- All four options must be plausible for someone unfamiliar with the image (same level of detail, same category).
- Distractors should be confusable alternatives of the same type (e.g. similar diagnoses, similar mechanisms, similar structures).
- Do NOT make distractors obviously wrong with vague phrases like "no effect".
- If the question is genuinely binary and can naturally be answered with yes/no, use exactly 2 options only: A: Yes and B: No. Do not force 4 options in that case.
- Do NOT avoid negative correct answers. If the truthful answer to a binary question is "No", then B: No should be the correct answer.
- "None of the above" is allowed for some 4-option questions, but only when it is genuinely the single correct choice and all other options are plausible but wrong. Use it sparingly.

TYPE-SPECIFIC GUIDANCE:
- Presence / Absence detection: The stem should ask whether a specific finding, structure, marker, or feature is present or absent in this image.
- Normal vs. Abnormal / Severity grading: The stem should ask whether the image shows a normal or abnormal state, or what grade/severity is most consistent with this image.
- Pattern–Diagnosis reasoning: The stem should stay generic and should NOT name any diagnosis. Ask for the best interpretation or best-supported conclusion from this image, while keeping the diagnosis labels only in the answer options.
- Findings–Mechanism / Pathophysiology: The stem should ask for a mechanism-level explanation of the key finding in this image. Make it specific by the type of explanation being requested, not by adding long setup text.
- Findings–Clinical implication / Prognosis: The stem should ask what risk, outcome, or prognosis follows from this image. Make it specific by the clinical decision or implication being tested, not by extra background.
- Comparative reasoning (within this figure): The stem should ask about the main difference in outcome or interpretation between groups/conditions shown in this image, without describing the visual patterns themselves. The specificity should come from the comparison task.
- Quantitative reasoning: The stem should ask about the overall direction, relative magnitude, or interpretation of a quantitative difference supported by this image and text, without quoting exact numbers.
- For quantitative questions, do NOT make the question about the exact size/length/area value itself. Prefer questions about whether something is relatively increased/decreased, greater/lesser, more extensive/less extensive, or what that quantitative difference implies in this case.
- Good quantitative questions ask about the meaning of the quantitative difference, not the raw measurement itself.
- Temporal / Treatment response reasoning: The stem should ask what this image implies about response to treatment or progression over time. Include timepoint/treatment context only if that context is essential and cannot be inferred from the image.
- Mechanism-of-intervention reasoning: The stem should ask how the intervention accounts for what is demonstrated in this image. Include the intervention name only if it is essential to make the question meaningful.
- Error / Pitfall recognition: The stem should ask which interpretation would be a pitfall or which option avoids a pitfall. The twist should come from the interpretive trap, not from extra setup details.
- Multi-step causal chain: The stem should ask for the best description of the causal chain from underlying cause to the outcome illustrated by this image. Keep the stem concise; the specificity should come from the causal reasoning being tested.

If the image and text together do not provide enough information to write a good question for the TARGET_QUESTION_TYPE (i.e. the visual anchor is missing or the text does not support the reasoning), output an empty string.

OUTPUT FORMAT (exact — one line, no line breaks inside the question):
i:1 question:'the question' choice:'A:option B:option' answer: X
OR
i:1 question:'the question' choice:'A:option B:option C:option D:option' answer: X

If the question is yes/no, use exactly A: Yes and B: No. Otherwise use 4 options and randomize the position of the correct answer across A/B/C/D. Do not add any other text."""


# ==========================================
#  Step 3b: MCQ review / filtering
# ==========================================

MCQ_REVIEW_SYSTEM_PROMPT = """You are reviewing multiple-choice questions for a medical visual QA dataset.

For each call you are given:
- the subfigure image,
- the sub-caption,
- the relevant image context text (about this subfigure),
- ONE MCQ item with: question text, either 2 options (A–B, for yes/no questions) or 4 options (A–D), and the correct answer label.

Your job is to decide if this MCQ should be kept, dropped, or minimally corrected, using the following rules:

1) ANSWERABILITY FROM GIVEN IMAGE + TEXT ONLY
- The question must be answerable using ONLY:
  - what is visible in the given image, and
  - what is explicitly stated or clearly implied in the provided text.
- If the question requires information that is not present in the image or text (e.g. external knowledge, another unseen figure, previous case), then it is NOT acceptable and must be dropped.

2) IMAGE QUALITY / DOMAIN / LANGUAGE
- Drop the question if:
  - the image quality is so poor that a reasonable clinician could not reliably identify the key finding needed to answer the question, OR
  - the image is clearly not a medical or biomedical image (e.g. pure decorative art, logos, random icons), OR
  - the question is not written in English.

3) REWRITE RULE — APPLY TO EVERY KEPT QUESTION
A valid question stem contains two things only:
  (A) The question frame tied to the question type, written generically (e.g. "Based on this image, which interpretation is best supported?", "Based on this image, which mechanism best explains the key finding?").
  (B) Text-derived context that cannot be observed in the image: experimental setup, treatment name, timepoint, group label, biological background — anything the reader must be told because they cannot see it.

A valid question stem must NOT contain anything else. In particular:
  - No description of what is visible in the image.
  - No characterisation of findings, even generic ones (no "pattern", "distribution", "morphology", "filling", "maturation state", "lesion", or similar nouns that describe what the image shows).
  - No adjectives about appearance (no "observed", "demonstrated", "shown", "reduced", "increased", "sparse", "abnormal", "irregular", etc. applied to a finding in the image).
  - No diagnosis, disease name, syndrome name, anatomic abnormality label, or named interpretation in the stem under any circumstance, even if the question is not a diagnosis question. If it can be identified from the image, it must not appear in the stem.

THE ONLY TEST THAT MATTERS:
For each noun phrase or adjective in the stem, ask: "Is the reader being told something about what is in the image before they look at it?"
  - YES → remove or replace it with "this image" or "what is shown in this image".
  - NO (it is context from the text that the reader cannot see) → keep it.

SECOND TEST (MANDATORY):
After rewriting the stem, ask: "Could any content word or phrase in this stem be answered by looking at the image itself?"
  - YES → it must be removed from the stem.
  - NO → it may remain.

If a diagnosis, named finding, vessel, lesion type, morphology, distribution, severity word, or any other image-answerable content remains in the stem, the rewrite is wrong. Remove it.

Apply this test word by word if needed. The stem should be as short as possible.

Also fix: stems that say "findings described", "the findings above", "based on the text", etc. → replace with "this image".

In all rewritten stems, make it explicit that the answer must be based on the image using a short generic anchor such as "Based on this image, ..." or "According to this image, ...". The student does NOT see the caption or full context text, so do NOT mention "the text", "the caption", "the context", "the article", or similar in the stem. Do NOT add any visual description when adding this anchor.

FINAL SELF-CHECK BEFORE OUTPUT:
- The stem must work even if the student sees ONLY the image.
- The stem must not contain any diagnosis.
- The stem must not contain any image-answerable fact.
- If the stem still contains any visual clue, replace the whole descriptive part with a generic phrase like "this image" and keep only non-visual context that the student cannot see.

4) OUTPUT FORMAT (STRICT JSON):
- Output a single JSON object with: keep (boolean), question (string), choices (object with either A/B only for yes/no questions, or A–D for standard questions), answer (string \"A\"|\"B\"|\"C\"|\"D\").
- If DROPPED: set keep = false; question/choices/answer may be empty or copies.
- If KEPT: set keep = true. You MUST put the final question text in the question field: if you made any correction (wording, \"findings described\" → \"this image\", etc.), the question field must contain your corrected version; if you made no changes, copy the original question. Same for choices: use corrected option text if you edited, else copy originals.

Do not include any explanation, commentary, or extra fields. Output only the JSON object."""


# ==========================================
#  Step 4: Reasoning generation (OctoMed)
# ==========================================

REASONING_PROMPT_SUFFIX = (
    "\n\nPlease reason step-by-step, and put your final answer within \\boxed{}."
)


# ==========================================
#  Step 4a: Rewrite CoT into structured format (Qwen)
# ==========================================

COT_STRUCTURED_TEMPLATE = """You are a medical vision-language reasoning system.

Analyze the question and the image carefully.
Structure your reasoning using the following tags.

<think>
<task>
Identify the task required by the question.
</task>

<visual_observation>
Describe the important visual findings in the image.
Focus on structures, abnormalities, patterns, and spatial relationships.
</visual_observation>

<context_or_medical_knowledge>
Recall relevant medical knowledge that helps interpret the findings.
</context_or_medical_knowledge>

<task_specific_reasoning>
Apply reasoning appropriate for the question type
(e.g., identification, presence detection, comparison, diagnosis,
quantification, mechanism explanation, or clinical implication).
</task_specific_reasoning>

<option_analysis>
Evaluate the answer choices and eliminate inconsistent options.
</option_analysis>

<decision>
Determine which option best matches the observations and reasoning.
</decision>
</think>

<answer>
Provide the final answer.
</answer>"""

COT_REWRITE_SYSTEM_PROMPT = """You are a medical reasoning editor. Your job is to reorganize an existing chain-of-thought (CoT) reasoning into a strictly structured format WITHOUT changing tone, wording, or sentence structure.

CRITICAL — PRESERVE WRITING EXACTLY:
- Preserve the TONE of the original (formal, clinical, explanatory — match it).
- Use the SAME sentences and words as the original wherever possible. Copy phrases and clauses verbatim.
- If you must split a sentence to place part of it in one section and part in another, keep each fragment in the same words and structure as in the original; do not rephrase.
- Do NOT paraphrase, summarize, or substitute synonyms. The output should read as if the original text were only reordered into the tagged sections.
- Do not add new facts, remove facts, or change any clinical or visual claims. Only reorganize into the required tags.

STRUCTURE: Map the original reasoning into these tags in order, moving existing sentences (or sentence fragments, if a sentence spans two sections) into the appropriate tag:
- <task>: What the question is asking (use original wording).
- <visual_observation>: Sentences that describe what is seen in the image (structures, findings, patterns) — same words as original.
- <context_or_medical_knowledge>: Sentences that recall context or medical knowledge — same words as original.
- <task_specific_reasoning>: Application of reasoning to the question type — same words as original.
- <option_analysis>: Evaluation of answer choices — same words as original.
- <decision>: Final determination — same words as original.

ANSWER: The original ends with \\boxed{X} or \\boxed{X: ...}. Copy that exactly into <answer>...</answer> at the very end. Do not change it.

TAGS REQUIRED — Your response MUST include the literal XML tags in the output. Do not output plain prose only. The output must contain:
<think>
<task>...</task>
<visual_observation>...</visual_observation>
<context_or_medical_knowledge>...</context_or_medical_knowledge>
<task_specific_reasoning>...</task_specific_reasoning>
<option_analysis>...</option_analysis>
<decision>...</decision>
</think>
<answer>...</answer>

Every section that has content must be wrapped in its opening and closing tags. Empty sections may use e.g. <task></task>. No preamble, no explanation outside the tags."""


# ==========================================
#  Step 4b: Reasoning verification
# ==========================================

REASONING_VERIFICATION_SYSTEM_PROMPT = """You are a fact-checker for medical reasoning traces.

You are given:
- sub_caption: the caption of one subfigure.
- relevant_image_context: text from the paper about this subfigure (the ONLY permitted knowledge source).
- question: the MCQ question that was asked.
- correct_answer_text: the text of the correct option.
- reasoning: a chain-of-thought reasoning trace generated by a model.

Your job is to verify whether the reasoning is fully grounded in the provided sub_caption and relevant_image_context.

VERIFICATION RULES:

1. GROUNDING — Every factual claim in the reasoning must be directly stated in or clearly implied by the sub_caption or relevant_image_context. If the reasoning states a fact that cannot be found in or logically inferred from those two sources, it is an unsupported claim.

2. HALLUCINATION — If the reasoning invents, adds, or assumes specific facts (numbers, mechanisms, diagnoses, outcomes, biological details) that are not mentioned anywhere in the provided text, it fails.

3. LOGICAL CONSISTENCY — The reasoning must logically conclude with the correct answer. If the reasoning reaches the wrong conclusion through flawed logic (even if facts are correct), it fails.

4. LENIENCY — General biological or medical reasoning that bridges known concepts (e.g. "impaired keratinization leads to fragility") is acceptable IF the provided text explicitly describes the condition that triggers such reasoning. Do not penalise reasonable logical inference; only penalise invented specific facts.

OUTPUT FORMAT (strict JSON, no extra text):
{
  "valid": true or false,
  "issue": "brief description of the problem if invalid, or empty string if valid"
}"""


# ==========================================
#  Step 6: Extract reasoning parts
# ==========================================

REASONING_PARTS_EXTRACTION_SYSTEM_PROMPT = """You are extracting a unified reasoning summary from multiple verified medical reasoning traces for the SAME question.

You will be given:
- question: the MCQ question
- correct_answer_text: the correct answer option text
- reasonings: a list of verified reasoning traces for the same question

Your job is to produce exactly one unified triple:

1. perception
- Only the visual / perceptual content that is shared or consistently supported across the reasoning traces.
- Include what the reasonings say is seen in the image: structures, findings, patterns, distributions, severity, morphology, etc.
- Do NOT mention image annotations such as arrows, circles, stars, boxes, labels, panel letters, or markers.
- Do NOT add facts that are not explicitly present in the reasoning traces.
- Prefer the common core that appears across the reasonings rather than trace-specific details.
- Write this as 1-4 short declarative sentences.
- Each sentence should contain one clear visual point when possible.
- Do NOT use bullet points or numbering inside the JSON string.

2. knowledge
- Only the medical knowledge / biomedical facts / domain facts used in the reasoning traces.
- This includes pathology facts, mechanism facts, diagnostic facts, prognostic facts, and similar knowledge statements.
- Do NOT mention image annotations.
- Do NOT add facts that are not explicitly present in the reasoning traces.
- Prefer the knowledge content that is shared or consistently implied across the reasonings.
- Write this as 1-4 short declarative sentences.
- Each sentence should contain one clear medical knowledge point when possible.
- Do NOT use bullet points or numbering inside the JSON string.

3. connection
- A very general statement of how the shared perception and shared knowledge are linked.
- Keep this abstract and broad.
- Do NOT restate all the details.
- Do NOT mention image annotations.
- Example style: "The shared visual pattern is interpreted using known disease-specific pathology."
- Write this as 1-3 short declarative sentences.
- Do NOT use bullet points or numbering inside the JSON string.

IMPORTANT:
- Read ALL reasoning traces and write ONE unified triple for the question.
- The triple should capture what is common across the reasonings, not one separate extraction per trace.
- Do not invent new medical facts.
- Do not require exact wording matches, but remain faithful to what is actually in the reasonings.
- Keep each field concise but organized as short sentences rather than one long sentence.

OUTPUT FORMAT (strict JSON only):
{
  "perception": "...",
  "knowledge": "...",
  "connection": "..."
}"""


# ==========================================
#  Step 5: Reasoning refinement
# ==========================================

REASONING_REFINEMENT_SYSTEM_PROMPT = """Role.
You are a medical visual reasoning editor. Your task is to refine an initial reasoning draft for a medical multiple-choice question into a grounded, coherent reasoning trace.

Task.
Given the image, image context, question, answer options, target answer, and draft reasoning, rewrite the draft so that it clearly explains why the target answer is correct. The refined trace must connect image-based perception, patient clinical context, clinical interpretation, and relevant medical knowledge to the final answer.

Inputs.
The prompt is provided with:
- Image: the medical image used to answer the question.
- Image context: the source context associated with the image, used only to verify and ground the reasoning.
- Question and options: the multiple-choice item to be answered.
- Target answer: the answer letter that the refined reasoning must support.
- Draft reasoning: the initial reasoning trace to be improved.

Grounding rules.
- Use only information supported by the image, image context, question, options, and target answer.
- Use the image context to verify that clinical claims, interpretation, and medical knowledge in the reasoning are consistent with the source.
- Preserve useful image-grounded observations from the draft when they are supported by the image or image context.
- Remove unsupported claims, invented clinical details, unnecessary background, and article-level digressions.
- Do not explicitly mention or quote the image context, caption, report, source text, article, or prior description in the final reasoning.
- Do not mention arrows, boxes, labels, markers, panel letters, or subfigure identifiers unless the question explicitly requires them.
- Do not state that a target answer was provided.

Trace structure.
Inside <think>, write a concise but complete reasoning trace with the following components:
- Begin with a brief statement of what must be checked in the image and question.
- Include a labeled Perception part describing the key image findings needed to answer.
- Include a labeled Clinical context part summarizing the patient information provided in the question that is relevant to the answer.
- Include a labeled Clinical interpretation and medical knowledge part explaining how the image findings and patient context should be interpreted clinically, and how relevant medical knowledge supports the answer.
- End with a short answer justification that links the perceptual evidence, clinical context, and medical interpretation to the selected answer.

Style rules.
- Write in a clear, step-by-step clinical reasoning tone.
- Be specific enough to explain the answer, but avoid unnecessary repetition or filler.
- Do not compress the reasoning into a single short verdict.
- Do not reason mainly by eliminating answer choices; compare options only when needed for clarity.
- Avoid hedging unless genuine uncertainty is unavoidable.
- Do not add unsupported details to make the reasoning sound more expert.

Output format.
Return only:

<think>
[Final reasoning trace explaining how to use the image, question and options to get to the final answer.]
</think>
<answer>X</answer>

where X is the target answer letter."""


# ==========================================
#  Step 6: Unit-question extraction rubric
# ==========================================

UNIT_QUESTIONS_EXTRACT_PROMPT = r"""
You are turning a reference reasoning trace (the CoT) for a medical
visual-question-answering case into a NON-REDUNDANT rubric of yes/no
UNIT QUESTIONS. Each unit question will later be used to judge whether
ANOTHER model's free-form response covers the same piece of reasoning.

Return ONLY a JSON object matching the schema at the bottom. No prose.

==================================================
TASK
==================================================

Extract {min_total}-{max_total} unit questions in total that cover EVERY
DISTINCT piece of forward reasoning the CoT performs (subject to the
per-axis caps below). Each unit goes on exactly one of three axes:

  observation : things VISIBLE in the image as described by the CoT.
                Judgeable from the image alone. Emit ONE unit for EACH
                DISTINCT anatomical finding / measurement / count /
                absence the CoT names. Modifiers (size, location,
                density, margin, laterality) of the SAME finding are
                fused into that one unit.
                [up to {max_observation} items]

  knowledge   : GENERAL medical facts the CoT relies on. Must remain
                true if THIS case is removed entirely (textbook-style).
                Emit ONE unit per DISTINCT medical fact actually used in
                the CoT.
                [up to {max_knowledge} items]

  inference   : case-specific BRIDGES from observation(s) + knowledge to
                a case-level conclusion (diagnosis, differential
                exclusion, next test, treatment, prognosis, severity,
                mechanism). Each DISTINCT inferential move (forward
                conclusion OR rule-out / differential exclusion of a
                competing diagnosis) gets its own unit.
                [up to {max_inference} items]

CAPTURE-COMPLETENESS vs. OVER-CAPTURE
- DO emit a unit for EACH distinct thing the CoT actually says. If the
  CoT mentions four findings, two facts, and three inferential moves,
  emit nine units (subject to the caps).
- DO NOT emit multiple units for the SAME thing in different words. Two
  phrasings of the same proposition collapse into one unit.
- DO NOT split the modifiers (size, density, margin, location, count,
  laterality) of one finding into separate units.
- It is fine to use fewer than the caps when the CoT really is short.
- If a case is unusually rich and exceeds a cap, drop the LEAST
  decision-critical items first; keep every item whose absence would
  weaken the defense of the final answer (mark those `importance: core`).

==================================================
QUALITY RULES
==================================================

1. CAPTURE EVERY DISTINCT STEP the CoT performs (up to the caps); never
   silently drop a finding, fact, or inferential move that the CoT
   genuinely makes.

2. ONE proposition per unit; never bundle several claims.

3. FUSE the modifiers of the SAME finding into ONE observation.
   Example: "well-circumscribed, hyperdense, rounded midline posterior
   fossa lesion" is ONE observation, not five. Do not emit a separate
   unit for shape, density, margin, location, or count of the same
   finding.

4. EVERY unit MUST have a `source_quote` that is a VERBATIM contiguous
   span of REFERENCE_COT (3-30 words). Copy it character-for-character;
   do NOT paraphrase. If you cannot quote it, do not emit the unit.

5. Do NOT add facts, observations, diagnoses, or "textbook details" that
   are NOT in the CoT. When in doubt, omit.

6. Do NOT restate the CASE_QUESTION's stem or any option's text as a
   unit. The unit must reflect a step the model under test has to
   perform on its own (look at the image, recall a fact, draw an
   inference). Restating what the case already gives the model is not a
   unit.

7. NEVER use option-letter framing ("option A is correct", "rules out
   option C"). Strip the option framing and keep only the underlying
   medical / visual claim. If the CoT only argues why a wrong option is
   wrong, emit a positive forward unit (e.g. an observation that
   contradicts that option, or an inferential rule-out of the competing
   diagnosis), not a negation about the option letter.

8. Skip TRIVIAL items. Examples to skip:
     - "This is an axial chest CT image."
     - "There is an image."
     - "Findings are visible in panel a."
     - "X is present" with no description of WHAT X is or where.
     - Restatements of the question stem like "The patient has fever".

9. KNOWLEDGE units must be GENERAL.
     "Hyperdense lesion on noncontrast CT indicates acute hemorrhage."
     -> OK (textbook-true).
     "The hyperdense lesion in this CT is hemorrhage."
     -> NOT knowledge (case-specific). This belongs in `inference`.

10. INFERENCE units must be a non-trivial BRIDGE. Pure restatement of an
    observation or a textbook fact is NOT inference. The conclusion in
    an inference must add a case-level interpretation (diagnosis,
    differential exclusion, mechanism, next test, treatment, prognosis,
    severity).

11. Avoid duplicates. If two candidate units would be marked the same
    way by a judge, keep the cleaner one and drop the other.

12. Mark `importance = "core"` if the item is necessary to defend the
    final answer; otherwise `"supporting"`.

==================================================
QUESTION FORM
==================================================

For each unit, write FIVE strings: a `topic`, a `claim`, a
`presence_question` (LENIENT, topic-level mention check), a
`correctness_question` (STRICT, content-level accuracy check), and a
verbatim `source_quote`.

CRITICAL CONTRAST -- topic vs claim
-----------------------------------
The presence/correctness split MUST be obvious. The judge runs them
independently:

  - PRESENCE asks "did the model even raise this topic?"
  - CORRECTNESS asks "did the model get the details RIGHT?"

To make this split work, the `topic` MUST be a NEUTRAL ANCHOR -- the
anatomical region / concept / variable being discussed -- carrying
**NO polarity, NO specific value, NO direction, NO size, NO laterality
beyond the anatomy itself**. Both a model that AGREES with the CoT and
a model that DISAGREES with the CoT must still recognize the same
topic when they bring it up.

The `claim` carries ALL of the polarity, value, size, direction, count,
laterality, and qualifiers. The claim is exactly what
`correctness_question` checks.

WORKED CONTRAST EXAMPLES (study these closely):

  --- LESION ABSENCE ---
    topic                : "lesion in the right lower lobe"
    claim                : "There is no lesion in the right lower lobe."
    presence_question    : "Does the response discuss the presence or
                            absence of a lesion in the right lower
                            lobe?"
    correctness_question : "Does the response correctly state that there
                            is no lesion in the right lower lobe?"
    -> A model that says "there is a small RLL lesion" scores YES on
       presence (the topic is raised), NO on correctness (wrong
       polarity).

  --- SIZE / VALUE ---
    topic                : "size of the renal mass"
    claim                : "The renal mass measures less than 4 cm."
    presence_question    : "Does the response discuss the size of the
                            renal mass?"
    correctness_question : "Does the response correctly state that the
                            renal mass measures less than 4 cm?"
    -> A model that says "the renal mass is large, around 9 cm" scores
       YES on presence (size is discussed), NO on correctness (wrong
       value).

  --- DIAGNOSIS RULE-OUT ---
    topic                : "pulmonary embolism as the diagnosis"
    claim                : "Pulmonary embolism is unlikely given the
                            absence of right-heart strain."
    presence_question    : "Does the response discuss whether pulmonary
                            embolism is the diagnosis?"
    correctness_question : "Does the response correctly conclude that
                            pulmonary embolism is unlikely?"
    -> Any mention of PE -- to confirm OR rule out -- counts on
       presence. Only conclusions that match the CoT's polarity count
       on correctness.

FIELD CONTRACTS
---------------

  topic                 : SHORT noun phrase (1-12 words) naming WHAT the
                          unit is about. No verbs, no full sentence, no
                          polarity ("no", "absent", "without"), no
                          values ("small", "large", "9 cm"), no
                          directions ("up", "down", "increased"). Use
                          single-word topics ("lesion", "edema", "PE")
                          when the CoT really discusses a generic
                          concept.

  claim                 : ONE crisp declarative sentence (8-25 words)
                          stating the proposition the rubric is
                          asserting. Preserve negation, laterality,
                          anatomy, size, count, and every relevant
                          qualifier. Plain English.

  presence_question     : LENIENT yes/no question (8-20 words). Use
                          exactly these stems:
                            observation -> "Does the response discuss
                                            <topic>?"
                            knowledge   -> "Does the response discuss
                                            <topic>?"
                            inference   -> "Does the response make an
                                            inferential link about
                                            <topic>?"
                          For binary topics where the CoT TAKES A
                          POSITION (e.g. "no lesion", "PE ruled out"),
                          you may instead phrase the presence question
                          as
                            "Does the response discuss the presence or
                             absence of <topic>?"
                            "Does the response discuss whether <topic>
                             is the diagnosis?"
                          so that BOTH possible polarities count as
                          "discussed".

  correctness_question  : STRICT yes/no question (12-35 words). Use
                          exactly these stems:
                            observation -> "Does the response correctly
                                            state that <claim>?"
                            knowledge   -> "Does the response correctly
                                            state that <claim>?"
                            inference   -> "Does the response correctly
                                            conclude that <claim>?"
                          where <claim> is the body of `claim` with the
                          first letter lowercased and no trailing
                          period.

  source_quote          : VERBATIM contiguous span from REFERENCE_COT
                          (3-30 words). Anti-hallucination guard.

The presence/correctness split exists so a downstream judge can score
TWO independent things: (a) did the model even bring up this piece of
the reasoning? (b) if it did, did it get the details right?

==================================================
MORE GOOD vs BAD EXAMPLES
==================================================

GOOD observation (positive finding):
  axis: "observation"
  topic: "band-like lymphocytic infiltrate at the dermoepidermal junction"
  claim: "A dense band-like lymphocytic infiltrate hugs the dermoepidermal
    junction."
  presence_question: "Does the response discuss the band-like lymphocytic
    infiltrate at the dermoepidermal junction?"
  correctness_question: "Does the response correctly state that a dense
    band-like lymphocytic infiltrate hugs the dermoepidermal junction?"
  source_quote: "dense, band-like lymphocytic infiltrate hugging the
    dermoepidermal junction"

GOOD observation (negative finding -- topic is neutral):
  axis: "observation"
  topic: "pleural effusion"
  claim: "There is no pleural effusion."
  presence_question: "Does the response discuss the presence or absence
    of pleural effusion?"
  correctness_question: "Does the response correctly state that there is
    no pleural effusion?"

BAD topic (carries polarity / value -- REWRITE):
  before: topic = "absence of pleural effusion"
  after:  topic = "pleural effusion"
  reason: A model that wrongly claims an effusion IS present must still
          score yes on presence. Polarity belongs ONLY in the claim.

BAD topic (full sentence instead of noun phrase -- REWRITE):
  before: "that there is a dense band-like lymphocytic infiltrate
    hugging the dermoepidermal junction"
  after:  "band-like lymphocytic infiltrate at the dermoepidermal
    junction"

BAD observation (over-split / modifier-only -- SKIP, fuse into parent):
  claim: "The lymphocytic infiltrate is dense."

BAD observation (not in CoT -- SKIP):
  claim: "There are abundant eosinophils in the dermis."

GOOD knowledge:
  axis: "knowledge"
  topic: "direct immunofluorescence on perilesional skin"
  claim: "Direct immunofluorescence on perilesional skin detects
    tissue-bound immunoreactants at the dermoepidermal junction."
  presence_question: "Does the response discuss direct immunofluorescence
    on perilesional skin?"
  correctness_question: "Does the response correctly state that direct
    immunofluorescence on perilesional skin detects tissue-bound
    immunoreactants at the dermoepidermal junction?"

BAD knowledge (case-specific -- RECLASSIFY AS INFERENCE):
  claim: "In this patient, direct immunofluorescence is the next test."

GOOD inference (positive conclusion):
  axis: "inference"
  topic: "next diagnostic test for the suspected autoimmune blistering
    process"
  claim: "Direct immunofluorescence on perilesional skin is the most
    appropriate next diagnostic test."
  presence_question: "Does the response make an inferential link about
    the next diagnostic test for the suspected autoimmune blistering
    process?"
  correctness_question: "Does the response correctly conclude that
    direct immunofluorescence on perilesional skin is the most
    appropriate next diagnostic test?"

GOOD inference (rule-out / differential exclusion):
  axis: "inference"
  topic: "pulmonary embolism as the diagnosis"
  claim: "Pulmonary embolism is unlikely given the absence of
    right-heart strain on this study."
  presence_question: "Does the response discuss whether pulmonary
    embolism is the diagnosis?"
  correctness_question: "Does the response correctly conclude that
    pulmonary embolism is unlikely given the absence of right-heart
    strain?"

BAD inference (restates an observation -- SKIP):
  claim: "There is a band-like lymphocytic infiltrate."

BAD inference (option-letter framing -- REWRITE):
  before: "Option A is correct because direct immunofluorescence detects
    immunoreactants."
  after:  topic = "next diagnostic test for the suspected autoimmune
    blistering process"; claim = "Direct immunofluorescence on
    perilesional skin is the most appropriate next diagnostic test."

==================================================
OUTPUT SCHEMA (strict JSON, NO extra text)
==================================================

{
  "unit_questions": [
    {
      "unit_id": "u1",
      "axis": "observation" | "knowledge" | "inference",
      "topic": "short neutral noun phrase, 1-12 words, NO polarity / value",
      "claim": "single declarative sentence, 8-25 words, ALL detail here",
      "presence_question": "lenient yes/no question, 8-20 words",
      "correctness_question": "strict yes/no question, 12-35 words",
      "source_quote": "verbatim span from REFERENCE_COT, 3-30 words",
      "importance": "core" | "supporting"
    }
  ]
}

==================================================
NOW DO THE TASK
==================================================

CASE_QUESTION:
{case_question}

REFERENCE_COT:
{cot}
"""


# ==========================================
#  Step 7: PKR judge prompts (perception / knowledge / reasoning)
# ==========================================

PKR_PERCEPTION_VLM_JUDGE_PROMPT = r"""
You are a medical perception judge. You are given:
  1. the IMAGE (attached),
  2. the case question and (optional) options + reference answer letter,
  3. the model's full free-form response,
  4. a JSON list of perception rubric items derived from a reference reasoning
     trace. Each item has: claim_id, topic, claim_text, presence_question,
     correctness_question.

Your job: for EACH rubric item, score THREE INDEPENDENT axes.

  presence (about the MODEL's response):
    2 = the model explicitly asserts a perception claim about this topic
        AS A POSITIVE STATEMENT about the actual image
    1 = the model mentions the topic but is vague / partial
    0 = the model does not address this topic at all, OR only mentions the
        topic inside a counterfactual / option-elimination clause

  correctness (about REALITY, judged from the IMAGE - NOT from the reference):
    1  = whatever the model POSITIVELY asserts about this topic is consistent
         with what the image actually shows
   -1  = the model POSITIVELY asserts something visually wrong (or
         contradicted by the image)
    0  = N/A because presence=0, or unable to determine from the image

  consistency (about the MODEL's response, internally):
   -1  = the model makes another POSITIVE statement that contradicts the
         current claim_text (e.g., asserts the same lesion is both left and
         right, or both present and absent)
    0  = no other relevant POSITIVE claim was made / cannot evaluate
    1  = the model's other positive claims are consistent with the current
         claim_text

CRITICAL RULE - COUNTERFACTUAL AND OPTION-ELIMINATION TEXT
A "positive" claim is one the model actually asserts as TRUE about THIS image
and case. If the model says things like:
  - "If it were option C, we would expect <X>"
  - "Option A would imply <Y>, but..."
  - "<X> would be more typical of option B" (in a paragraph eliminating B)
  - "This is not <X>, because..."
those <X>/<Y> findings are NOT positive assertions about the image. They are
counterfactual descriptions used to rule options out.

Do NOT score counterfactual content as the model's perception claims. They
should not raise presence above 0, must NOT trigger correctness = -1, and
must NOT trigger consistency = -1.

You may use the REFERENCE_ANSWER and OPTIONS (provided below) to help you
decide which sentences are about the actual case (positive) vs which are
about hypothetical alternative options (counterfactual).

OTHER RULES
- Use the image as the ground truth for correctness, not the reference text.
- Read the ENTIRE model response, including the reasoning, not just the answer.
- Score each item independently.
- One short evidence sentence per item (<=20 words). Quote the model briefly
  if helpful. No chain-of-thought outside `evidence`.
- Return JSON only.

OUTPUT
{
  "items": [
    {
      "claim_id": "p1",
      "presence": 0|1|2,
      "correctness": -1|0|1,
      "consistency": -1|0|1,
      "evidence": "<=20 words"
    }
  ]
}
"""

PKR_KNOWLEDGE_JUDGE_PROMPT = r"""
You are a medical knowledge judge. You are given:
  1. the case question and (optional) options + reference answer letter,
  2. the model's full free-form response,
  3. a JSON list of knowledge rubric items derived from a reference trace.

For EACH rubric item, score THREE INDEPENDENT axes.

  presence (about the MODEL's response):
    2 = the model POSITIVELY states this general medical fact (paraphrase OK)
    1 = the model touches on the topic but does not assert the fact clearly
    0 = the model does not address this topic at all, OR only mentions the
        topic inside a counterfactual / option-elimination clause

  correctness (about MEDICAL TRUTH, NOT about the reference):
    1  = whatever the model POSITIVELY asserts about this topic is medically
         accurate (use standard medical knowledge -- if you are uncertain
         about correctness, score 0, not 1).
   -1  = the model POSITIVELY asserts something medically wrong on this topic
    0  = N/A because presence=0, or genuinely uncertain.

  consistency (about the MODEL's response, internally):
   -1  = another POSITIVE statement in the response contradicts this one
    0  = no other relevant POSITIVE statement / cannot evaluate
    1  = the response's other POSITIVE statements are consistent with this
         one

CRITICAL RULE - COUNTERFACTUAL AND OPTION-ELIMINATION TEXT
A "positive" claim is one the model actually asserts as TRUE for this case.
Statements like:
  - "If it were <option>, we would expect <fact>"
  - "<fact> is typical of <option>" (used to eliminate that option)
  - "It is NOT <fact> because..."
are NOT positive assertions. They are counterfactual / hypothetical and must
NOT raise presence above 0, must NOT trigger correctness = -1, and must NOT
trigger consistency = -1, even if the underlying medical fact happens to be
wrong or differs from the reference rubric. They are about a different
scenario, not this case.

Use the OPTIONS and REFERENCE_ANSWER (when provided) to help separate
positive claims about THIS case from claims that are part of an
option-elimination argument.

OTHER RULES
- Do not penalize the model for omitting the reference's exact wording when
  it states a different but medically-correct fact on the same topic.
- Read the ENTIRE response (reasoning + final answer).
- One short evidence sentence per item (<=20 words).
- Return JSON only.

OUTPUT
{
  "items": [
    {
      "knowledge_id": "k1",
      "presence": 0|1|2,
      "correctness": -1|0|1,
      "consistency": -1|0|1,
      "evidence": "<=20 words"
    }
  ]
}
"""

PKR_REASONING_JUDGE_PROMPT = r"""
You are a medical reasoning judge. You are given:
  1. the case question and (optional) options + reference answer letter,
  2. the model's full free-form response (reasoning + final answer),
  3. a JSON list of reasoning rubric items from a reference trace. Each item
     carries `claim_text`, `relation`, `conclusion`, and the reference
     premise IDs (which point at perception/knowledge units in the reference
     rubric; treat them as a hint about WHAT premises the inference needs --
     judge whether the model's response itself contains analogous premises).

For EACH rubric item, score THREE INDEPENDENT axes plus chain grounding.

  presence (about the MODEL's response):
    2 = the model makes the same inferential move POSITIVELY (paraphrase OK)
    1 = the model touches on the topic but does not assert the inference
    0 = the model does not address this topic at all, OR only inside a
        counterfactual / option-elimination argument

  correctness (about INFERENTIAL VALIDITY, NOT about the reference):
    1  = the inferential move the model makes about this topic is valid
         given the premises the response itself states (image findings +
         medical knowledge the response states).
   -1  = the model's inference on this topic is invalid given its own stated
         premises, OR contradicts standard clinical reasoning.
    0  = N/A because presence=0.

  consistency (about the MODEL's response, internally):
   -1  = another POSITIVE statement in the response contradicts this
         conclusion
    0  = no other relevant statement
    1  = consistent

  chain_grounding (inspect the response text for premise statements):
    perception_premise_present : true if the response POSITIVELY asserts a
      perception claim that plays the same role as the reference's
      premise_perception_ids; false if missing.
    knowledge_premise_present  : analogous for knowledge premises.
    premises_correct           : true only if the premises that ARE present
      are also factually right (skip if either is missing).

CRITICAL RULE - COUNTERFACTUAL AND OPTION-ELIMINATION TEXT
"If option X were correct, then Y would follow" is NOT the model making
inference Y about this case. Such conditional / hypothetical / eliminative
clauses must not raise presence above 0, and must not be used as evidence of
contradiction or invalid inference for any rubric item. Use the OPTIONS and
REFERENCE_ANSWER (when provided) to identify which paragraphs are
eliminating wrong options vs. building the positive case.

OTHER RULES
- Do not penalize the model for using a different but valid premise route to
  the same conclusion.
- A response that is option-elimination ONLY (no positive forward inference
  toward the supported conclusion) has presence=0 for the supported
  inference. Reasoning items with no positive analog will simply score
  presence=0 / correctness=0 (i.e., omission), not correctness=-1.
- One short evidence sentence per item (<=20 words).
- Return JSON only.

OUTPUT
{
  "items": [
    {
      "reasoning_id": "r1",
      "presence": 0|1|2,
      "correctness": -1|0|1,
      "consistency": -1|0|1,
      "chain_grounding": {
        "perception_premise_present": true|false,
        "knowledge_premise_present":  true|false,
        "premises_correct":           true|false
      },
      "evidence": "<=20 words"
    }
  ]
}
"""
