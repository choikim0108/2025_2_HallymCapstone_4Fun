using UnityEngine;
using Controller;

public class SimplePlayerCamera : PlayerCamera
{
    protected override void Awake()
    {
        base.Awake();
        // 추가 세팅 필요시 여기에 작성
    }

    void LateUpdate()
    {
        if (m_Player == null) return;

        // 각도(라디안) 계산
        float yaw = m_Angles.y;
        float pitch = m_Angles.x;

        // 회전 쿼터니언 생성
        Quaternion rotation = Quaternion.Euler(pitch, yaw, 0f);

        // 카메라 위치 계산 (플레이어 기준 뒤쪽, 위쪽 오프셋)
        Vector3 offset = rotation * new Vector3(0, 0, -m_Distance);
        Vector3 targetPos = m_Player.position + Vector3.up * 2f + offset;

        m_Transform.position = targetPos;
        m_Transform.rotation = rotation;
    }
}