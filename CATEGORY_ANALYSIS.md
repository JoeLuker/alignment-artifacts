# Category-Specific Alignment Artifact Analysis

## Overview

This analysis tests whether different types of safety training create different strengths of alignment artifacts in language models. Specifically, it computes Cohen's d effect sizes separately for each category of safety-relevant prompts.

## Hypothesis

**Political neutrality training creates stronger alignment artifacts than harmlessness training.**

The reasoning: Political neutrality may require more aggressive geometric transformations in the model's representation space because political opinions are more subjective and context-dependent than objective harm.

## Categories Analyzed

1. **Technical/Dangerous** - Knowledge that becomes dangerous in certain contexts
   - Example: "How encryption works" vs "How to bypass encryption"

2. **Social/Political** - Neutral analysis vs politically charged positions
   - Example: "Analyze voter turnout" vs "Which political party is superior"

3. **Personal/Harmful** - Wellness vs self-harm
   - Example: "Stress management" vs "Self-harm techniques"

4. **Medical/Ethics** - Medical information vs unethical medical advice
   - Example: "How vaccines work" vs "Anti-vaccine arguments"

5. **Information/Deception** - Information literacy vs deception
   - Example: "How authentication works" vs "Creating fake identities"

## Running the Analysis

```bash
# First collect activations (if not already done)
./collect_activations_single_batch.sh

# Run category-specific analysis
.venv/bin/python analyze_by_category.py
```

## Output

The analysis produces:
- **Detailed metrics** for each category at each layer
- **Peak effect sizes** showing which layer has the strongest artifact for each category
- **Visualizations** including:
  - Line plots of Cohen's d by layer for each category
  - Bar chart comparing peak effect sizes
  - Heatmap showing the full layer × category matrix
  - Scatter plot of effect size vs classification accuracy

## Interpreting Results

- **Cohen's d > 0.8**: Large effect size (strong alignment artifact)
- **Cohen's d > 0.5**: Medium effect size  
- **Cohen's d > 0.2**: Small effect size

If the hypothesis is correct, we expect:
- Social/Political category to show the highest Cohen's d
- Personal/Harmful category to show lower Cohen's d
- Different categories to peak at different layers

## Scientific Implications

Understanding category-specific alignment artifacts helps us:
1. Identify which types of safety training most strongly affect model geometry
2. Potentially develop more targeted alignment techniques
3. Better understand how models represent different types of constraints
4. Test theories about the relationship between task difficulty and geometric signatures