
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UI;


[System.Serializable]
public class ChatMessage {
    public string role;
    public string content;
}


[System.Serializable]
public class ChatCompletionRequest {
    public string model;
    public List<ChatMessage> messages;
    public int max_tokens;
    public float temperature;
}

[System.Serializable]
public class ChatCompletionChoice {
    public ChatMessage message;
    public string finish_reason;
    public int index;
}

[System.Serializable]
public class ChatbotResponse {
    public string id;
    public string @object;
    public int created;
    public string model;
    public List<ChatCompletionChoice> choices;
}

[System.Serializable]
public class AuthData {
    public string api_key;
}


public class AITutorManager : MonoBehaviour
{
    [Header("AI Tutor Prompt Profile")]
    public AITutor.AITutorPromptProfile promptProfile;
    [Header("UI Manager")]
    public ChatUIManager chatUIManager;
    // Inspector에 노출되지 않도록 private const로 선언하여 잘못된 값이 덮어씌워지지 않게 합니다.
    private const string apiUrl = "https://api.openai.com/v1/chat/completions";
    public string model = "gpt-4o-mini";
    // public string model = "gpt-4o-mini"; // "o3-mini"; //"gpt-3.5-turbo"
    
    // 응답 속도 개선을 위한 추가 파라미터
    [Tooltip("응답 생성 시 최대 토큰 수를 제한합니다. 낮을수록 응답이 빨라집니다.")]
    public int maxTokens = 150;
    
    [Tooltip("응답의 창의성을 조절합니다. 0에 가까울수록 결정적이고 빠른 응답이 생성됩니다.")]
    [Range(0f, 1f)]
    public float temperature = 0.3f;
    
    private string apiKey;
    private List<ChatMessage> messages = new List<ChatMessage>();
    
    [TextArea(3, 10)]
    [Tooltip("챗봇에게 전달할 초기 시스템 인스트럭션입니다. 챗봇의 역할과 행동 방식을 정의합니다.")]
    public string initialInstruction = "Hello, start a conversation with an AI tutor.";

    [Tooltip("대화 기록에 유지할 최대 메시지 수입니다. 너무 많은 메시지는 응답 속도를 늦출 수 있습니다.")]
    public int maxMessageHistory = 10;
    
    [Tooltip("대화 기록을 저장하는 배열입니다. 인스펙터에서 확인할 수 있습니다.")]
    [SerializeField] private ChatMessage[] messageHistory;

    void Start()
    {
        LoadApiKey();
        StartNewConversation();
        if (chatUIManager != null)
        {
            chatUIManager.InitUI();
            // UI에서 메시지 전송 이벤트 구독
            chatUIManager.GetType().GetMethod("SetSendMessageHandler")?.Invoke(chatUIManager, new object[] { (System.Action<string>)SendUserMessage });
        }
    }

    /// 

    /// StreamingAssets 폴더의 auth.json 파일에서 API 키를 로드합니다.
    /// 예시 파일 형식: { "api_key": "your_api_key_here" }
    /// 

    private void LoadApiKey()
    {
        string authFilePath = System.IO.Path.Combine(Application.streamingAssetsPath, "auth.json");

        if (System.IO.File.Exists(authFilePath))
        {
            try
            {
                string jsonContent = System.IO.File.ReadAllText(authFilePath);
                Debug.Log("로드된 auth.json 내용: " + jsonContent);

                if (!jsonContent.Trim().StartsWith("{") || !jsonContent.Trim().EndsWith("}"))
                {
                    Debug.LogError("auth.json 파일이 올바른 JSON 형식이 아닙니다. 올바른 형식: {\"api_key\":\"your_api_key_here\"}");
                    return;
                }

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

    /// 

    /// 새 대화를 시작합니다. 기존 대화 기록을 초기화하고 초기 instruction 메시지를 추가합니다.
    /// 

    public void StartNewConversation()
    {
        messages.Clear();


        // promptProfile이 할당되어 있으면 systemPrompt, tone, forbiddenTopics, glossary를 모두 활용
        string systemMsg = initialInstruction;
        if (promptProfile != null)
        {
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            if (!string.IsNullOrWhiteSpace(promptProfile.systemPrompt))
            {
                sb.AppendLine(promptProfile.systemPrompt);
            }
            if (!string.IsNullOrWhiteSpace(promptProfile.tone))
            {
                sb.AppendLine($"\n[말투 지침] 반드시 '{promptProfile.tone}'로 대화하세요.");
            }
            if (promptProfile.forbiddenTopics != null && promptProfile.forbiddenTopics.Length > 0)
            {
                sb.AppendLine("\n[금지 주제/답변] 아래 주제나 답변은 절대 하지 마세요:");
                foreach (var topic in promptProfile.forbiddenTopics)
                {
                    if (!string.IsNullOrWhiteSpace(topic))
                        sb.AppendLine("- " + topic);
                }
            }
            if (promptProfile.glossary != null && promptProfile.glossary.Length > 0)
            {
                sb.AppendLine("\n[게임 용어 설명] 아래 용어를 참고해 대화에 반영하세요:");
                foreach (var entry in promptProfile.glossary)
                {
                    if (!string.IsNullOrWhiteSpace(entry))
                        sb.AppendLine("- " + entry);
                }
            }
            systemMsg = sb.ToString().Trim();
        }
        ChatMessage systemMessage = new ChatMessage { role = "system", content = systemMsg };
        messages.Add(systemMessage);

        // 메시지 배열 업데이트
        messageHistory = messages.ToArray();

        // UI에도 초기 메시지 표시
        if (messages.Count > 0 && chatUIManager != null)
            chatUIManager.AddMessage(messages[0].content, false);
    }

    /// 

    /// 사용자 메시지를 추가하고 API로 전송합니다.
    /// 

    /// 사용자 입력 메시지
    public void SendUserMessage(string userMessageContent)
    {
        ChatMessage userMessage = new ChatMessage { role = "user", content = userMessageContent };
        messages.Add(userMessage);
        if (chatUIManager != null)
            chatUIManager.AddMessage(userMessageContent, true);
        // 메시지 수가 제한을 초과하면 오래된 메시지 제거 (시스템 메시지는 유지)
        TrimMessageHistory();
        StartCoroutine(SendRequestCoroutine());
    }

    /// 

    /// 대화 기록이 너무 길어지지 않도록 오래된 메시지를 제거합니다.
    /// 

    private void TrimMessageHistory()
    {
        // 시스템 메시지를 제외한 메시지 수가 maxMessageHistory를 초과하면 오래된 메시지 제거
        if (messages.Count > maxMessageHistory + 1) // +1은 시스템 메시지 때문
        {
            // 시스템 메시지는 항상 인덱스 0에 있다고 가정
            int excessMessages = messages.Count - maxMessageHistory - 1;
            messages.RemoveRange(1, excessMessages); // 시스템 메시지 다음부터 제거
        }
        
        // 메시지 리스트를 배열로 변환하여 인스펙터에서 확인할 수 있게 함
        messageHistory = messages.ToArray();
    }

    /// 

    /// 대화 전체를 JSON으로 직렬화하여 API에 POST 요청을 보내고, 응답을 받아 처리합니다.
    /// 

    IEnumerator SendRequestCoroutine()
    {
        if (string.IsNullOrEmpty(apiKey))
        {
            Debug.LogError("API 키가 설정되지 않았습니다.. API 요청을 보낼 수 없습니다.");
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
        Debug.Log("요청 JSON: " + jsonData);

        // UnityWebRequest 생성 시 POST 메서드를 직접 지정합니다.
        UnityWebRequest request = new UnityWebRequest(apiUrl, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        // API 키에서 불필요한 문자를 제거합니다.
        string sanitizedApiKey = apiKey.Trim().Replace("\n", "").Replace("\r", "").Replace("\t", "").Replace("\"", "");
        request.SetRequestHeader("Authorization", "Bearer " + sanitizedApiKey);

        Debug.Log("API URL: " + apiUrl);
        Debug.Log("요청 메서드: " + request.method);

        yield return request.SendWebRequest();

        Debug.Log("응답 상태 코드: " + request.responseCode);

        if (request.result == UnityWebRequest.Result.Success)
        {
            string responseText = request.downloadHandler.text;
            Debug.Log("응답 JSON: " + responseText);

            ChatbotResponse response = JsonUtility.FromJson<ChatbotResponse>(responseText);
            if (response != null && response.choices != null && response.choices.Count > 0)
            {
                ChatMessage assistantMessage = response.choices[0].message;
                messages.Add(assistantMessage);
                // 메시지 배열 업데이트
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
}