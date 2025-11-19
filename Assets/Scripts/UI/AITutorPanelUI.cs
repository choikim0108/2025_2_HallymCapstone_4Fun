using UnityEngine;
using UnityEngine.UI;
using System;
using TMPro;

namespace UI
{
    public class AITutorPanelUI : MonoBehaviour
    {
        //public Button openButton;
        public GameObject chatPanelRoot;
        public ScrollRect chatScrollRect;
        public TMP_InputField inputField;
        public Button sendButton;
        public GameObject scrollbarVertical;

        public event Action<string> OnSendMessage;

        void Start()
        {
            //if (openButton != null)
                //openButton.onClick.AddListener(TogglePanel);
            if (sendButton != null)
                sendButton.onClick.AddListener(SendMessageToAI);
            if (chatPanelRoot != null)
                chatPanelRoot.SetActive(false);
            // AITutorPanelUI 오브젝트를 비활성화하면 안 됨 (스크립트가 작동을 멈춤)
            //this.gameObject.SetActive(false);
        }

        public void TogglePanel()
        {
            if (chatPanelRoot != null)
            {
                bool newState = !chatPanelRoot.activeSelf;
                chatPanelRoot.SetActive(newState);
                // AITutorPanelUI 오브젝트 자체도 같이 활성화/비활성화
                //this.gameObject.SetActive(newState);
                if (scrollbarVertical != null)
                    scrollbarVertical.SetActive(newState); // 강제로 동기화
                // 패널이 열릴 때만 InputField에 포커스
                if (newState && inputField != null)
                {
                    inputField.ActivateInputField();
                }
            }
        }
        // [추가] CursorController가 현재 패널 상태를 확인할 수 있도록 함
        public bool IsPanelActive()
        {
            if (chatPanelRoot == null)
            {
                return false;
            }
            return chatPanelRoot.activeSelf;
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
