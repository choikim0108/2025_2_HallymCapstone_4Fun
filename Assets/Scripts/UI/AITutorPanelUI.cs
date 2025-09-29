using UnityEngine;
using UnityEngine.UI;
using System;
using TMPro;

namespace UI
{
    public class AITutorPanelUI : MonoBehaviour
    {
        public Button openButton;
        public GameObject chatPanelRoot;
        public ScrollRect chatScrollRect;
        public TMP_InputField inputField;
        public Button sendButton;
        public GameObject scrollbarVertical;

        public event Action<string> OnSendMessage;

        void Start()
        {
            if (openButton != null)
                openButton.onClick.AddListener(TogglePanel);
            if (sendButton != null)
                sendButton.onClick.AddListener(SendMessageToAI);
            if (chatPanelRoot != null)
                chatPanelRoot.SetActive(false);
        }

        public void TogglePanel()
        {
            if (chatPanelRoot != null)
            {
                bool newState = !chatPanelRoot.activeSelf;
                chatPanelRoot.SetActive(newState);
                if (scrollbarVertical != null)
                    scrollbarVertical.SetActive(newState); // 강제로 동기화
                // 패널이 열릴 때만 InputField에 포커스
                if (newState && inputField != null)
                {
                    inputField.ActivateInputField();
                }
            }
        }

        public void SendMessageToAI()
        {
            if (inputField != null && !string.IsNullOrWhiteSpace(inputField.text))
            {
                OnSendMessage?.Invoke(inputField.text);
                inputField.text = string.Empty;
            }
        }

        public void ScrollToBottom()
        {
            if (chatScrollRect != null)
                chatScrollRect.verticalNormalizedPosition = 0f;
        }
    }
}
