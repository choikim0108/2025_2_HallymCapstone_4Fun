using UnityEngine;

namespace AITutor
{
    [CreateAssetMenu(fileName = "AITutorPromptProfile", menuName = "AI Tutor/Prompt Profile", order = 1)]
    public class AITutorPromptProfile : ScriptableObject
    {
        [TextArea(3, 10)]
        [Tooltip("게임에 맞는 AI 튜터의 역할, 말투, 행동 지침 등을 작성하세요.")]
        public string systemPrompt = "너는 이 게임의 AI 튜터야. 플레이어가 게임을 더 잘 이해하고 즐길 수 있도록 친절하고 명확하게 설명해줘. 게임 규칙, 팁, 힌트, 전략, 그리고 플레이어의 질문에 맞는 답변을 제공해. 플레이어가 막히거나 어려워하면 격려와 함께 구체적인 도움을 줘. 게임 내 용어와 세계관을 잘 반영해서 대화해.";

        [Tooltip("튜터의 말투(예: 존댓말, 반말, 캐릭터화 등)")]
        public string tone = "밝고 친근한 존댓말";

        [Tooltip("게임 내에서 AI 튜터가 피해야 할 답변이나 금지어 등")]
        public string[] forbiddenTopics;

        [Tooltip("게임 내에서 자주 쓰는 용어 및 설명")]
        public string[] glossary;
    }
}
