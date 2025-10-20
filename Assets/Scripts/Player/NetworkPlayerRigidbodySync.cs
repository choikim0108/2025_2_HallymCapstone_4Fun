using UnityEngine;
using Photon.Pun;

public class NetworkPlayerRigidbodySync : MonoBehaviour
{
    private PhotonView photonView;
    private Rigidbody rb;

    void Awake()
    {
        photonView = GetComponent<PhotonView>();
        rb = GetComponent<Rigidbody>();
        if (photonView != null && rb != null)
        {
            // 내 플레이어만 물리 연산, 상대방은 위치만 동기화
            rb.isKinematic = !photonView.IsMine;
        }
    }
}
