using UnityEngine;

namespace UI
{
    public class ChatUIManager : MonoBehaviour
    {
        public AITutorPanelUI tutorPanelUI;
        public GameObject chatMessagePrefab;
        public Transform messageListRoot;

        public void InitUI()
        {
            if (tutorPanelUI != null)
            {
                tutorPanelUI.openButton.gameObject.SetActive(true);
                tutorPanelUI.chatPanelRoot.SetActive(false);
                tutorPanelUI.OnSendMessage += OnSendMessageFromUI;
            }
        }

        public void AddMessage(string message, bool isUser)
        {
            if (chatMessagePrefab != null && messageListRoot != null)
            {
                GameObject msgObj = Instantiate(chatMessagePrefab, messageListRoot);
                var textComp = msgObj.GetComponentInChildren<TMPro.TextMeshProUGUI>();
                if (textComp != null)
                    textComp.text = message;
                // 사용자/AI 메시지에 따라 스타일 분기 가능
            }
            if (tutorPanelUI != null)
                tutorPanelUI.ScrollToBottom();
        }

        public void ClearMessages()
        {
            if (messageListRoot != null)
            {
                foreach (Transform child in messageListRoot)
                {
                    Destroy(child.gameObject);
                }
            }
        }

        // 외부에서 메시지 전송 핸들러를 등록할 수 있도록 지원
        public void SetSendMessageHandler(System.Action<string> handler)
        {
            if (tutorPanelUI != null)
            {
                tutorPanelUI.OnSendMessage -= OnSendMessageFromUI;
                tutorPanelUI.OnSendMessage += handler;
            }
        }

        // UI에서 전송버튼 클릭 시 호출됨
        private void OnSendMessageFromUI(string userMessage)
        {
            // AITutorManager에서 이 메서드를 구독해 실제 메시지 전송 처리
        }
    }
}