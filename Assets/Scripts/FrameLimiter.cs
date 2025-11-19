using UnityEngine;

public class FrameLimiter : MonoBehaviour
{
    void Awake()
    {
        // 모니터 주사율과 상관없이 강제로 60프레임으로 고정
        QualitySettings.vSyncCount = 0; 
        Application.targetFrameRate = 60; 
    }
}