using UnityEngine;
using UnityEngine.UI;

public class SetupPanelController : MonoBehaviour
{
    public GameObject setupPanel;
    public Button setupPanelOpenButton;  // Button 타입 유지
    public Button setupPanelCloseButton; // Button 타입 유지

    void Start()
    {
        // 초기화
        ClosePanel();

        // 버튼 이벤트 안전하게 연결
        if (setupPanelOpenButton != null)
        {
            setupPanelOpenButton.onClick.RemoveAllListeners();
            setupPanelOpenButton.onClick.AddListener(OpenPanel);
        }

        if (setupPanelCloseButton != null)
        {
            setupPanelCloseButton.onClick.RemoveAllListeners();
            setupPanelCloseButton.onClick.AddListener(ClosePanel);
        }
    }

    public void OpenPanel()
    {
        if (setupPanel != null) setupPanel.SetActive(true);
        // Button 타입이므로 .gameObject를 통해 SetActive 사용
        if (setupPanelOpenButton != null) setupPanelOpenButton.gameObject.SetActive(false);
        if (setupPanelCloseButton != null) setupPanelCloseButton.gameObject.SetActive(true);
    }

    public void ClosePanel()
    {
        if (setupPanel != null) setupPanel.SetActive(false);
        if (setupPanelOpenButton != null) setupPanelOpenButton.gameObject.SetActive(true);
        if (setupPanelCloseButton != null) setupPanelCloseButton.gameObject.SetActive(false);
    }
}