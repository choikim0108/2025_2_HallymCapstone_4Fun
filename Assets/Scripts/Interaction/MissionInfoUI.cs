using UnityEngine;
using TMPro; // TextMeshPro 사용
using System.Collections;

public class MissionInfoUI : MonoBehaviour
{
    public static MissionInfoUI Instance;

    [Header("UI Components")]
    public GameObject missionPanel;      // 미션 정보가 담긴 패널 (켜고 끄기 위함)
    public TextMeshProUGUI missionText;  // 실제 텍스트 (예: "폭발물 박스를 배달하세요")

    void Awake()
    {
        // 싱글톤 설정
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }

        // 시작할 때는 패널 숨기기
        if (missionPanel != null)
            missionPanel.SetActive(false);
    }

    // 외부에서 이 함수를 호출하여 미션을 화면에 띄움
    public void ShowMission(string boxName)
    {
        if (missionPanel == null || missionText == null) return;

        missionPanel.SetActive(true); // 패널 켜기

        // 박스 이름에 따라 표시할 텍스트 변경 (필요에 따라 한글화)
        string displayName = GetKoreanName(boxName);
        missionText.text = $"<color=yellow>새 임무:</color>\n{displayName}을(를) 배달하세요!";
        
        // 일정 시간 뒤에 끄고 싶다면 아래 주석 해제
        // StopAllCoroutines();
        // StartCoroutine(HidePanelAfterDelay(5f)); 
    }

    // 미션 완료 시 숨기는 함수 (나중에 미션 성공 로직에서 호출 가능)
    public void HideMission()
    {
        if (missionPanel != null)
            missionPanel.SetActive(false);
    }

    // 영문 박스 이름을 한글로 예쁘게 변환
    private string GetKoreanName(string resourceName)
    {
        switch (resourceName)
        {
            case "Box_Normal": return "일반 박스";
            case "Box_Fragile": return "파손주의 박스";
            case "Box_Heavy": return "무거운 박스";
            case "Box_Explosive": return "폭발물 박스";
            case "Box_Frozen": return "냉동 박스";
            default: return resourceName; // 목록에 없으면 영어 그대로 표시
        }
    }

    private IEnumerator HidePanelAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        HideMission();
    }
}