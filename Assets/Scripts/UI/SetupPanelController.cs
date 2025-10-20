using UnityEngine;
using UnityEngine.UI;

public class SetupPanelController : MonoBehaviour
{
    public GameObject setupPanel;
    public Button setupPanelOpenButton;
    public Button setupPanelCloseButton;

    void Start()
    {
        // 시작 시 SetupPanel은 비활성화, OpenButton은 활성화, CloseButton은 비활성화
        setupPanel.SetActive(false);
        setupPanelOpenButton.gameObject.SetActive(true);
        setupPanelCloseButton.gameObject.SetActive(false);

        setupPanelOpenButton.onClick.AddListener(OpenPanel);
        setupPanelCloseButton.onClick.AddListener(ClosePanel);
    }

    public void OpenPanel()
    {
        setupPanel.SetActive(true);
        setupPanelOpenButton.gameObject.SetActive(false);
        setupPanelCloseButton.gameObject.SetActive(true);
    }

    public void ClosePanel()
    {
        setupPanel.SetActive(false);
        setupPanelOpenButton.gameObject.SetActive(true);
        setupPanelCloseButton.gameObject.SetActive(false);
    }
}
