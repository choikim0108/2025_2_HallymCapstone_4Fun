using UnityEngine;
using UI; // AITutorPanelUI를 사용하기 위해 네임스페이스 추가

public class CursorController : MonoBehaviour
{
    // UI 컨트롤러 참조
    public AITutorPanelUI aiTutorPanel;
    public RoomInfoController roomInfoController;
    // 'isCursorLocked' 변수 제거 (이제 상태는 UI 패널이 결정)

    // NPC 대화 시스템 UI 오브젝트 (NPC와 대화할 때 켜지는 패널)
    public GameObject npcDialogueSys;

    void Start()
    {
        // 게임이 시작되면 일단 커서를 잠금 상태(FPS 모드)로 설정
        SetCursorState(true);
    }

    void Update()
    {
        // 엔터 키로 AITutorPanel 토글
        if (Input.GetKeyDown(KeyCode.Return))
        {
            if (aiTutorPanel != null)
            {
                aiTutorPanel.TogglePanel();
            }
        }

        // 탭 키로 RoomInfoController의 패널 토글
        if (Input.GetKeyDown(KeyCode.Tab))
        {
            if (roomInfoController != null)
            {
                roomInfoController.ToggleTabPanel();
            }
        }

        // 마우스 우클릭으로 RoomInfoController의 패널 토글
        if (Input.GetMouseButtonDown(1))
        {
            if (aiTutorPanel != null)
            {
                aiTutorPanel.TogglePanel();
            }
        }

        // --- 커서 상태 관리 ---
        // 어떤 UI 패널이든 하나라도 활성화되어 있는지 확인
        bool isUIVisible = false;
        if (aiTutorPanel != null && aiTutorPanel.IsPanelActive())
        {
            isUIVisible = true;
        }
        if (roomInfoController != null && roomInfoController.IsPanelActive())
        {
            isUIVisible = true;
        }
        if (npcDialogueSys != null && npcDialogueSys.activeSelf)
        {
            isUIVisible = true;
        }
        // UI가 보이면 커서 잠금 해제, UI가 모두 꺼지면 커서 잠금
        // isUIVisible == true  -> isLocked = false
        // isUIVisible == false -> isLocked = true
        SetCursorState(!isUIVisible);
    }

    /// <summary>
    /// 커서의 상태를 설정하는 함수
    /// </summary>
    /// <param name="isLocked">true: FPS 모드, false: 메뉴 모드</param>
    void SetCursorState(bool isLocked)
    {
        if (isLocked)
        {
            // --- FPS 모드 ---
            Cursor.visible = false;
            Cursor.lockState = CursorLockMode.Locked;
        }
        else
        {
            // --- 메뉴/UI 모드 ---
            Cursor.visible = true;
            Cursor.lockState = CursorLockMode.None;
        }
    }
}