using UnityEngine;

public class MissionStart : MonoBehaviour
{
	// 미션 실패 처리 함수
	public void MissionFail(string reason)
	{
		Debug.Log($"[MissionStart] Mission Failed: {reason}");
		// 실패 UI, 사운드, 네트워크 동기화 등 추가 구현 가능
	}
}
