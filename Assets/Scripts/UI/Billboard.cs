using UnityEngine;

public class Billboard : MonoBehaviour
{
    private Camera mainCamera;

    void Start()
    {
        // 성능을 위해 메인 카메라를 캐싱합니다.
        mainCamera = Camera.main; 
    }

    // LateUpdate는 모든 Update가 끝난 후 호출되므로 카메라 이동 후에 UI 방향을 설정하기 좋습니다.
    void LateUpdate()
    {
        if (mainCamera == null)
        {
            // 혹시 메인 카메라를 못찾았다면 다시 시도
            mainCamera = Camera.main;
            if (mainCamera == null) return; // 그래도 없으면 종료
        }

        // UI가 카메라와 같은 방향을 바라보게 합니다.
        // transform.LookAt(mainCamera.transform); // 이 방법은 UI가 뒤집힐 수 있습니다.

        // 더 안정적인 방법: 카메라의 전방 벡터의 반대 방향을 바라보게 합니다.
        transform.LookAt(transform.position + mainCamera.transform.rotation * Vector3.forward,
                         mainCamera.transform.rotation * Vector3.up);
    }
}