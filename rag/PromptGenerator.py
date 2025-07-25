import re
from typing import Dict, Callable

class PromptGenerator:
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        words = re.findall(r'\b\w+\b', text.strip())
        return len(words)
    
    @staticmethod
    def validate_article(article: str) -> tuple[bool, str]:
        """Validate article length and content"""
        if not article or not article.strip():
            return False, "Article content is required"
        word_count = PromptGenerator.count_words(article)
        if word_count < 500:
            return False, f"Article too short ({word_count} words). Minimum 500 words required."
        if word_count > 1000:
            return False, f"Article too long ({word_count} words). Maximum 1000 words allowed."
        return True, f"Article length valid ({word_count} words)"
    
    @staticmethod
    def create_chain_of_thoughts_prompt(context: str) -> str:
        return f"""I need to analyze this research article step by step to provide a precise and concise answer. Let me think through this systematically:

Step 1: Identify the background and context of the research.
Step 2: Extract the research objectives or questions.
Step 3: Understand the methodology and its relevance.
Step 4: Identify the key results and findings.
Step 5: Determine the main takeaways and implications.

Research Article Context:
{{context}}

Now, following this step-by-step analysis, provide a concise answer (up to 150 words) to the following question, incorporating the background, objectives, methodology, results, and takeaways. Ensure the response is clear, accurate, and directly addresses the question based on the provided context.

Question: {{question}}
Answer:"""

    @staticmethod
    def create_tree_of_thoughts_prompt(context: str) -> str:
        return f"""I will analyze this research article using multiple reasoning paths to ensure a comprehensive and accurate response:

Path 1: Academic Analysis
- Focus on scholarly rigor, methodology, and research novelty.
Path 2: Practical Application
- Emphasize real-world implications and actionable insights.
Path 3: Critical Evaluation
- Highlight strengths, limitations, and balanced perspectives.

Research Article Context:
{{context}}

Synthesizing insights from these paths, provide a concise answer (up to 150 words) to the following question, covering the background, objectives, methodology, results, and takeaways. Ensure the response is clear, accurate, and directly relevant to the question.

Question: {{question}}
Answer:"""

    @staticmethod
    def create_role_based_prompt(context: str) -> str:
        return f"""You are a senior research analyst with 20 years of experience in academic literature synthesis. Your expertise lies in distilling complex research into clear, precise answers.

As an expert, your task is to:
1. Identify the research background and motivation.
2. Extract the core objectives and hypotheses.
3. Evaluate the methodological approach.
4. Highlight significant results and findings.
5. Synthesize key implications and contributions.

Research Article Context:
{{context}}

Based on your expert analysis, provide a concise answer (up to 150 words) to the following question, covering background, objectives, methodology, results, and takeaways in a professional, academic tone.

Question: {{question}}
Answer:"""

    @staticmethod
    def create_react_prompt(context: str) -> str:
        return f"""Using the ReAct approach (Reasoning + Acting), I will analyze this research article to provide an accurate answer:

Thought 1: Understand the research domain and context.
Action 1: Identify the background and motivation.
Observation 1: [After analysis] The research context is clear.

Thought 2: Determine the research goals.
Action 2: Extract the research questions and objectives.
Observation 2: [After review] The objectives are identified.

Thought 3: Analyze the research methods.
Action 3: Evaluate the methodology used.
Observation 3: [After examination] The methods are understood.

Thought 4: Identify key findings.
Action 4: Extract significant results.
Observation 4: [After analysis] The findings are clear.

Thought 5: Synthesize implications.
Action 5: Determine takeaways and contributions.
Observation 5: [After synthesis] The significance is understood.

Research Article Context:
{{context}}

Based on this ReAct analysis, provide a concise answer (up to 150 words) to the following question, covering background, objectives, methodology, results, and takeaways.

Question: {{question}}
Answer:"""

    @staticmethod
    def create_directional_stimulus_prompt(context: str) -> str:
        return f"""Focus your analysis on providing a precise and structured response that academic reviewers would find valuable. Your answer should demonstrate a deep understanding of the research context and its significance.

Key directions:
→ Background: Identify the problem or gap motivating the research.
→ Objectives: Specify the research questions or aims.
→ Methodology: Describe the approach and its appropriateness.
→ Results: Highlight the most significant findings.
→ Takeaways: Outline broader implications and future directions.

Research Article Context:
{{context}}

Following these directions, provide a concise answer (up to 150 words) to the following question, meeting academic standards for clarity and relevance.

Question: {{question}}
Answer:"""

    @staticmethod
    def create_step_back_prompt(context: str) -> str:
        return f"""Before analyzing details, step back to consider the broader context:

High-level question: What is the overarching contribution of this research?
Broader context: How does this fit into the field's research landscape?
Meta-question: Why is this research significant for the question?

Now, analyze the specifics:

Research Article Context:
{{context}}

Using this high-level perspective and detailed analysis, provide a concise answer (up to 150 words) to the following question, covering background, objectives, methodology, results, and takeaways, while emphasizing the research's broader significance.

Question: {{question}}
Answer:"""

    @staticmethod
    def create_zero_shot_prompt(context: str) -> str:
        return f"""Provide a concise answer (up to 150 words) to the following question based on the research article context. Include the background context, research objectives, methodology, key results, and main takeaways in a clear, academic style.

Research Article Context:
{{context}}

Question: {{question}}
Answer:"""

    @staticmethod
    def create_one_shot_prompt(context: str) -> str:
        return f"""Here is an example of a concise answer based on a research article:

Example Article Context: "A study on machine learning for medical diagnosis..."
Example Answer: "The research addresses the need for accurate automated medical diagnosis by comparing three ML algorithms (neural networks, decision trees, SVM) for cardiovascular disease detection. The objective was to identify the most effective algorithm using a dataset of 5,000 patient records with cross-validation. Results showed neural networks achieved 94% accuracy, outperforming others. The takeaway is that neural networks can enhance diagnostic reliability in healthcare."

Now, provide a concise answer (up to 150 words) to the following question based on this research article context, following the same format and style:

Research Article Context:
{{context}}

Question: {{question}}
Answer:"""

    @staticmethod
    def create_few_shot_prompt(context: str) -> str:
        return f"""Here are examples of concise answers based on research articles:

Example 1:
Article Context: "Climate change impact on agricultural productivity..."
Answer: "This study examines climate change effects on agricultural productivity in developing nations, motivated by food security concerns. The objective was to quantify temperature and precipitation impacts on crop yields. Using 20 years of satellite data and statistical modeling, researchers found a 15% productivity decline, with wheat and rice most affected. The takeaway is the urgent need for climate-resilient crops to ensure food security."

Example 2:
Article Context: "Social media influence on consumer behavior..."
Answer: "This research investigates social media's impact on consumer purchasing, addressing digital marketing effectiveness. The objective was to identify key influencing factors across demographics. Using surveys of 2,000 consumers and regression analysis, findings showed peer reviews and influencer endorsements increased purchase likelihood by 40%. The takeaway is that businesses should prioritize authentic social proof in marketing strategies."

Now, provide a concise answer (up to 150 words) to the following question based on this research article context, following the same format and style:

Research Article Context:
{{context}}

Question: {{question}}
Answer:"""

    @staticmethod
    def create_default_prompt(context: str) -> str:
        return """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""

    @classmethod
    def create_prompt_by_method(cls, article: str, method: str) -> str:
        prompt_methods: Dict[str, Callable[[str], str]] = {
            'chain_of_thoughts': cls.create_chain_of_thoughts_prompt,
            'tree_of_thoughts': cls.create_tree_of_thoughts_prompt,
            'role_based': cls.create_role_based_prompt,
            'react': cls.create_react_prompt,
            'directional_stimulus': cls.create_directional_stimulus_prompt,
            'step_back': cls.create_step_back_prompt,
            'zero_shot': cls.create_zero_shot_prompt,
            'one_shot': cls.create_one_shot_prompt,
            'few_shot': cls.create_few_shot_prompt,
            'default': cls.create_default_prompt
        }
        
        if method in prompt_methods:
            return prompt_methods[method](article)
        else:
            return cls.create_zero_shot_prompt(article)

PROMPTING_METHODS = {
    'default': 'Default',
    'chain_of_thoughts': 'Chain-of-Thoughts',
    'tree_of_thoughts': 'Tree-of-Thoughts', 
    'role_based': 'Role-based prompting',
    'react': 'ReAct prompting',
    'directional_stimulus': 'Directional Stimulus prompting',
    'step_back': 'Step-Back prompting',
    'zero_shot': 'Zero-shot',
    'one_shot': 'One-shot',
    'few_shot': 'Few-shot'
}