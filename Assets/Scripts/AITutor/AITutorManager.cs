using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UI;

public class AITutorManager : MonoBehaviour
{
    [Header("AI Tutor Prompt Profile")]
    public AITutor.AITutorPromptProfile promptProfile;

    [Header("UI Manager")]
    public ChatUIManager chatUIManager;

    [Header("Dynamic Context")]
    [Tooltip("현재 진행 중인 미션 이름")]
    public string currentMission = "Delivering Various Packages to NPCs";

    [Tooltip("학생의 영어 레벨 (Beginner, Intermediate, Advanced)")]
    public string studentLevel = "Beginner";

    [Tooltip("현재 게임 내 위치")]
    public string currentLocation = "Post Office";

    private const string apiUrl = "https://api.openai.com/v1/chat/completions";
    public string model = "gpt-4o";

    [Tooltip("응답 생성 시 최대 토큰 수")]
    public int maxTokens = 250;

    [Tooltip("응답의 창의성 (0에 가까울수록 일관적)")]
    [Range(0f, 1f)]
    public float temperature = 0.2f;

    [Tooltip("대화 기록 최대 유지 수")]
    public int maxMessageHistory = 15;

    private string apiKey;
    private List<ChatMessage> messages = new List<ChatMessage>();

    [SerializeField] private ChatMessage[] messageHistory;

    void Start()
    {
        LoadApiKey();
        StartNewConversation();
        if (chatUIManager != null)
        {
            chatUIManager.InitUI();
            chatUIManager.GetType().GetMethod("SetSendMessageHandler")?.Invoke(chatUIManager, new object[] { (System.Action<string>)SendUserMessage });
        }
    }

    private void LoadApiKey()
    {
        string authFilePath = System.IO.Path.Combine(Application.streamingAssetsPath, "auth.json");

        if (System.IO.File.Exists(authFilePath))
        {
            try
            {
                string jsonContent = System.IO.File.ReadAllText(authFilePath);
                AuthData authData = JsonUtility.FromJson<AuthData>(jsonContent);
                if (authData != null && !string.IsNullOrEmpty(authData.api_key))
                {
                    apiKey = authData.api_key;
                    Debug.Log("API 키가 성공적으로 로드되었습니다.");
                }
                else
                {
                    Debug.LogError("auth.json 파일에서 API 키를 찾을 수 없습니다.");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError("auth.json 파일 파싱 중 오류 발생: " + e.Message);
            }
        }
        else
        {
            Debug.LogError("auth.json 파일을 찾을 수 없습니다. 경로: " + authFilePath);
        }
    }

    public void StartNewConversation()
    {
        messages.Clear();

        string systemPrompt = BuildSystemPrompt();
        ChatMessage systemMessage = new ChatMessage { role = "system", content = systemPrompt };
        messages.Add(systemMessage);

        // Few-shot examples 추가
        if (promptProfile != null && promptProfile.examples != null)
        {
            foreach (var example in promptProfile.examples)
            {
                if (!string.IsNullOrWhiteSpace(example.userMessage) && !string.IsNullOrWhiteSpace(example.assistantResponse))
                {
                    messages.Add(new ChatMessage { role = "user", content = example.userMessage });
                    messages.Add(new ChatMessage { role = "assistant", content = example.assistantResponse });
                }
            }
        }

        messageHistory = messages.ToArray();

        if (chatUIManager != null)
            chatUIManager.AddMessage("Hello!, I'm 4Fun AI Hint System. 안녕하세요! 4Fun의 AI 힌트 시스템입니다.", false);
    }

    private string BuildSystemPrompt()
    {
        if (promptProfile == null)
            return "당신은 메타버스 속 영어 학습 도우미입니다. 힌트 시스템으로 명명되어있으며, 학생 플레이어에게 힌트를 제공하면 됩니다.";

        StringBuilder sb = new StringBuilder();
    
// 게임 컨텍스트 강제 규칙 최우선 배치
        sb.AppendLine("### ABSOLUTE PRIORITY ###");
        sb.AppendLine("This is a FICTIONAL GAME. ALL questions are about IN-GAME activities.");
        sb.AppendLine("DO NOT provide real-world advice about actual delivery services, courier companies, or postal procedures.");
        sb.AppendLine("ONLY discuss game NPCs, missions, English learning within the Post Office game map.");
        sb.AppendLine();

        // 1. 역할
        if (!string.IsNullOrWhiteSpace(promptProfile.role))
        {
            sb.AppendLine(promptProfile.role);
            sb.AppendLine();
        }

        // 2. 응답 원칙
        if (!string.IsNullOrWhiteSpace(promptProfile.responseGuidelines))
        {
            sb.AppendLine("## 응답 원칙");
            sb.AppendLine(promptProfile.responseGuidelines);
            sb.AppendLine();
        }

        // 3. 힌트 전략
        if (!string.IsNullOrWhiteSpace(promptProfile.hintStrategy))
        {
            sb.AppendLine("## 힌트 제공 방식");
            sb.AppendLine(promptProfile.hintStrategy);
            sb.AppendLine();
        }

        // 4. 금지 사항
        if (!string.IsNullOrWhiteSpace(promptProfile.restrictions))
        {
            sb.AppendLine("## 금지 사항");
            sb.AppendLine(promptProfile.restrictions);
            sb.AppendLine();
        }

        // 5. 게임 용어
        if (promptProfile.gameTerms != null && promptProfile.gameTerms.Length > 0)
        {
            sb.AppendLine("## 게임 용어");
            foreach (var term in promptProfile.gameTerms)
            {
                if (!string.IsNullOrWhiteSpace(term.term))
                    sb.AppendLine($"- {term.term}: {term.description}");
            }
            sb.AppendLine();
        }

        // 6. 동적 컨텍스트
        if (!string.IsNullOrWhiteSpace(promptProfile.dynamicContextTemplate))
        {
            string context = promptProfile.dynamicContextTemplate
                .Replace("{missionName}", currentMission)
                .Replace("{studentLevel}", studentLevel)
                .Replace("{currentLocation}", currentLocation);
            sb.AppendLine(context);
        }

        return sb.ToString().Trim();
    }

    public void SendUserMessage(string userMessageContent)
    {
        ChatMessage userMessage = new ChatMessage { role = "user", content = userMessageContent };
        messages.Add(userMessage);
        if (chatUIManager != null)
            chatUIManager.AddMessage(userMessageContent, true);

        TrimMessageHistory();
        StartCoroutine(SendRequestCoroutine());
    }

    private void TrimMessageHistory()
    {
        // 시스템 메시지 + Few-shot examples + 최근 대화만 유지
        int systemAndExamplesCount = 1 + (promptProfile != null && promptProfile.examples != null ? promptProfile.examples.Length * 2 : 0);
        int maxTotal = systemAndExamplesCount + maxMessageHistory;

        if (messages.Count > maxTotal)
        {
            int excessMessages = messages.Count - maxTotal;
            messages.RemoveRange(systemAndExamplesCount, excessMessages);
        }

        messageHistory = messages.ToArray();
    }

    IEnumerator SendRequestCoroutine()
    {
        if (string.IsNullOrEmpty(apiKey))
        {
            Debug.LogError("API 키가 설정되지 않았습니다.");
            yield break;
        }

        ChatCompletionRequest requestObj = new ChatCompletionRequest
        {
            model = model,
            messages = messages,
            max_tokens = maxTokens,
            temperature = temperature
        };

        string jsonData = JsonUtility.ToJson(requestObj);
        UnityWebRequest request = new UnityWebRequest(apiUrl, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        string sanitizedApiKey = apiKey.Trim().Replace("\n", "").Replace("\r", "").Replace("\t", "").Replace("\"", "");
        request.SetRequestHeader("Authorization", "Bearer " + sanitizedApiKey);

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string responseText = request.downloadHandler.text;
            ChatbotResponse response = JsonUtility.FromJson<ChatbotResponse>(responseText);
            if (response != null && response.choices != null && response.choices.Count > 0)
            {
                ChatMessage assistantMessage = response.choices[0].message;
                messages.Add(assistantMessage);
                messageHistory = messages.ToArray();
                if (chatUIManager != null)
                    chatUIManager.AddMessage(assistantMessage.content, false);
            }
        }
        else
        {
            Debug.LogError("API 요청 오류: " + request.error);
            Debug.LogError("응답 내용: " + request.downloadHandler.text);
        }
    }

    // 동적 컨텍스트 업데이트 메서드
    public void UpdateContext(string mission, string level, string location)
    {
        currentMission = mission;
        studentLevel = level;
        currentLocation = location;

        // 시스템 메시지 갱신
        if (messages.Count > 0)
        {
            messages[0].content = BuildSystemPrompt();
            messageHistory = messages.ToArray();
        }
    }
}

[System.Serializable]
public class AuthData
{
    public string api_key;
}

[System.Serializable]
public class ChatMessage
{
    public string role;
    public string content;
}

[System.Serializable]
public class ChatCompletionRequest
{
    public string model;
    public List<ChatMessage> messages;
    public int max_tokens;
    public float temperature;
}

[System.Serializable]
public class ChatbotResponse
{
    public List<Choice> choices;
}

[System.Serializable]
public class Choice
{
    public ChatMessage message;
}
