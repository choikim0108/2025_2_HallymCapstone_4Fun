using UnityEngine;

namespace UI
{
    public class ChatUIManager : MonoBehaviour
    {
    public AITutorPanelUI tutorPanelUI;
    // public GameObject chatMessagePrefab; // 기존 한 줄을 아래 두 줄로 변경
    public GameObject ChatMessagePrefab_USER;      // 내가 보낸 메시지용 프리팹
    public GameObject ChatMessagePrefab_AI;   // 상대가 보낸 메시지용 프리팹
    public Transform messageListRoot;

        public void InitUI()
        {
            Debug.Log("[ChatUIManager] InitUI 호출됨");
            if (tutorPanelUI != null)
            {
                Debug.Log("[ChatUIManager] tutorPanelUI 존재, UI 초기화 진행");
                tutorPanelUI.openButton.gameObject.SetActive(true);
                tutorPanelUI.chatPanelRoot.SetActive(false);
                tutorPanelUI.OnSendMessage += OnSendMessageFromUI;
            }
            else
            {
                Debug.LogWarning("[ChatUIManager] tutorPanelUI가 할당되지 않음");
            }
        }

        /*10.02.09:28 수정 72~131 line
        public void AddMessage(string message, bool isUser)
        {
            Debug.Log($"[ChatUIManager] AddMessage 호출됨 | isUser: {isUser}, message: {message}");

            if (chatMessagePrefab != null && messageListRoot != null)
            {
                // AddMessage 내부
                GameObject msgObj = Instantiate(chatMessagePrefab, messageListRoot);
                RectTransform rt = msgObj.GetComponent<RectTransform>();
                if (rt != null)
                {
                    rt.anchorMin = new Vector2(0, 1);
                    rt.anchorMax = new Vector2(1, 1);
                    rt.pivot = new Vector2(0.5f, 1);
                    rt.offsetMin = new Vector2(0, rt.offsetMin.y);
                    rt.offsetMax = new Vector2(0, rt.offsetMax.y);
                }
                Debug.Log("[ChatUIManager] chatMessagePrefab 인스턴스화 완료");
                var textComp = msgObj.GetComponentInChildren<TMPro.TextMeshProUGUI>();
                if (textComp != null)
                {
                    textComp.text = message;
                    Debug.Log("[ChatUIManager] TextMeshProUGUI에 메시지 할당 완료");
                }
                else
                {
                    Debug.LogWarning("[ChatUIManager] TextMeshProUGUI 컴포넌트를 찾을 수 없음");
                }
                // 사용자/AI 메시지에 따라 스타일 분기 가능
            }
            else
            {
                Debug.LogWarning("[ChatUIManager] chatMessagePrefab 또는 messageListRoot가 null임");
            }
            if (tutorPanelUI != null)
            {
                tutorPanelUI.ScrollToBottom();
                Debug.Log("[ChatUIManager] ScrollToBottom 호출됨");
            }
        }*/


        public void AddMessage(string message, bool isUser)
        {
            Debug.Log($"[ChatUIManager] AddMessage 호출됨 | isUser: {isUser}, message: {message}");

            // 1. 어떤 프리팹을 사용할지 담을 임시 변수를 선언합니다.
            GameObject prefabToUse;

            // 2. isUser 값에 따라 사용할 프리팹을 결정합니다.
            if (isUser)
            {
                // isUser가 true이면, '내가 보낸 메시지' 프리팹을 선택합니다.
                prefabToUse = ChatMessagePrefab_USER;
            }
            else
            {
                // isUser가 false이면, '상대가 보낸 메시지' 프리팹을 선택합니다.
                prefabToUse = ChatMessagePrefab_AI;
            }

            // 3. 결정된 프리팹이 Unity 에디터에서 할당되었는지, 그리고 메시지가 표시될 부모(messageListRoot)가 있는지 확인합니다.
            if (prefabToUse != null && messageListRoot != null)
            {
                // 4. 위에서 결정된 프리팹(prefabToUse)으로 실제 게임 오브젝트를 생성합니다.
                GameObject msgObj = Instantiate(prefabToUse, messageListRoot);

                /*
                RectTransform rt = msgObj.GetComponent<RectTransform>();
                if (rt != null)
                {
                    rt.anchorMin = new Vector2(0, 1);
                    rt.anchorMax = new Vector2(1, 1);
                    rt.pivot = new Vector2(0.5f, 1);
                    rt.offsetMin = new Vector2(0, rt.offsetMin.y);
                    rt.offsetMax = new Vector2(0, rt.offsetMax.y);
                }
                */
                
                // 생성된 메시지 오브젝트에서 TextMeshProUGUI 컴포넌트를 찾아 메시지를 할당합니다.
                var textComp = msgObj.GetComponentInChildren<TMPro.TextMeshProUGUI>();
                if (textComp != null)
                {
                    textComp.text = message;
                    Debug.Log("[ChatUIManager] TextMeshProUGUI에 메시지 할당 완료");
                }
                else
                {
                    Debug.LogWarning("[ChatUIManager] TextMeshProUGUI 컴포넌트를 찾을 수 없음");
                }
            }
            else
            {
                // 프리팹이 할당되지 않았을 경우 경고 메시지를 표시합니다.
                Debug.LogWarning("[ChatUIManager] ChatMessagePrefab_USER 또는 ChatMessagePrefab_AI가 할당되지 않았습니다.");
            }

            if (tutorPanelUI != null)
            {
                tutorPanelUI.ScrollToBottom();
                Debug.Log("[ChatUIManager] ScrollToBottom 호출됨");
            }
        }

        public void ClearMessages()
        {
            Debug.Log("[ChatUIManager] ClearMessages 호출됨");
            if (messageListRoot != null)
            {
                foreach (Transform child in messageListRoot)
                {
                    Destroy(child.gameObject);
                }
                Debug.Log("[ChatUIManager] messageListRoot의 모든 자식 오브젝트 삭제 완료");
            }
            else
            {
                Debug.LogWarning("[ChatUIManager] messageListRoot가 null임");
            }
        }

        // 외부에서 메시지 전송 핸들러를 등록할 수 있도록 지원
        public void SetSendMessageHandler(System.Action<string> handler)
        {
            Debug.Log("[ChatUIManager] SetSendMessageHandler 호출됨");
            if (tutorPanelUI != null)
            {
                tutorPanelUI.OnSendMessage -= OnSendMessageFromUI;
                tutorPanelUI.OnSendMessage += handler;
                Debug.Log("[ChatUIManager] OnSendMessage 핸들러 등록 완료");
            }
            else
            {
                Debug.LogWarning("[ChatUIManager] tutorPanelUI가 null임");
            }
        }

        // UI에서 전송버튼 클릭 시 호출됨
        private void OnSendMessageFromUI(string userMessage)
        {
            Debug.Log($"[ChatUIManager] OnSendMessageFromUI 호출됨 | userMessage: {userMessage}");
            // AITutorManager에서 이 메서드를 구독해 실제 메시지 전송 처리
        }
    }
}