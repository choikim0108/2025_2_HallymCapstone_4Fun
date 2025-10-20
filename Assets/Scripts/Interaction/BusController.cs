using UnityEngine;
using Photon.Pun;
using System.Collections.Generic;
using System.Linq;

public class BusController : MonoBehaviourPun
{
    public float moveSpeed = 10f;
    public float turnSpeed = 50f;
    public Transform exitPoint; // 플레이어가 하차할 위치 (버스 옆 빈 오브젝트)

    private int driverViewID = -1;
    private List<int> passengerViewIDs = new List<int>();

    void Awake()
    {
        // exitPoint가 설정되지 않았다면 버스 위치를 기준으로 설정
        if (exitPoint == null)
        {
            exitPoint = new GameObject("ExitPoint").transform;
            exitPoint.SetParent(this.transform);
            exitPoint.localPosition = new Vector3(2.5f, 0, 0); // 버스 오른쪽 2.5m
        }
    }

    void Update()
    {
        // 운전자만 버스를 조종할 수 있음
        if (driverViewID != -1 && PhotonView.Find(driverViewID).IsMine)
        {
            HandleMovement();
        }
    }

    private void HandleMovement()
    {
        float moveInput = Input.GetAxis("Vertical");
        transform.Translate(Vector3.forward * moveInput * moveSpeed * Time.deltaTime);

        float turnInput = Input.GetAxis("Horizontal");
        transform.Rotate(Vector3.up * turnInput * turnSpeed * Time.deltaTime);
    }

    // [RPC - MasterClient Only] 플레이어의 탑승 요청을 처리
    [PunRPC]
    public void BoardRequestRPC(int playerViewID)
    {
        if (!PhotonNetwork.IsMasterClient) return;

        // 이미 탑승 중인 플레이어는 요청 무시
        if (driverViewID == playerViewID || passengerViewIDs.Contains(playerViewID)) return;

        if (driverViewID == -1) // 운전석이 비어있으면
        {
            photonView.RPC("SetDriverRPC", RpcTarget.All, playerViewID);
        }
        else // 운전석이 차있으면
        {
            photonView.RPC("AddPassengerRPC", RpcTarget.All, playerViewID);
        }
    }

    // [RPC - MasterClient Only] 플레이어의 하차 요청을 처리
    [PunRPC]
    public void GetOffRequestRPC(int playerViewID)
    {
        if (!PhotonNetwork.IsMasterClient) return;

        if (driverViewID == playerViewID) // 운전자가 내리는 경우
        {
            // 모든 탑승자도 함께 내리게 함
            List<int> allRiders = new List<int>(passengerViewIDs);
            allRiders.Add(driverViewID);

            foreach (var riderID in allRiders)
            {
                photonView.RPC("EjectPlayerRPC", RpcTarget.All, riderID);
            }

            // 버스 상태 초기화
            photonView.RPC("ResetBusRPC", RpcTarget.All);
        }
        else if (passengerViewIDs.Contains(playerViewID)) // 탑승자가 내리는 경우
        {
            photonView.RPC("EjectPlayerRPC", RpcTarget.All, playerViewID);
        }
    }

    // [RPC - All Clients] 운전자를 설정
    [PunRPC]
    private void SetDriverRPC(int playerViewID)
    {
        driverViewID = playerViewID;
        var playerView = PhotonView.Find(playerViewID);
        if (playerView != null && playerView.IsMine)
        {
            playerView.GetComponent<PlayerInteraction>().OnBoardedAsDriver(this.gameObject);
        }
    }

    // [RPC - All Clients] 탑승자를 추가
    [PunRPC]
    private void AddPassengerRPC(int playerViewID)
    {
        if (!passengerViewIDs.Contains(playerViewID))
        {
            passengerViewIDs.Add(playerViewID);
            var playerView = PhotonView.Find(playerViewID);
            if (playerView != null && playerView.IsMine)
            {
                playerView.GetComponent<PlayerInteraction>().OnBoardedAsPassenger(this.gameObject);
            }
        }
    }

    // [RPC - All Clients] 특정 플레이어를 하차시킴
    [PunRPC]
    private void EjectPlayerRPC(int playerViewID)
    {
        // 하차할 플레이어 목록에서 제거
        if (driverViewID == playerViewID) driverViewID = -1;
        if (passengerViewIDs.Contains(playerViewID)) passengerViewIDs.Remove(playerViewID);

        var playerView = PhotonView.Find(playerViewID);
        if (playerView != null && playerView.IsMine)
        {
            playerView.GetComponent<PlayerInteraction>().OnGetOffBus(exitPoint.position);
        }
    }

    // [RPC - All Clients] 버스의 모든 상태를 초기화
    [PunRPC]
    private void ResetBusRPC()
    {
        driverViewID = -1;
        passengerViewIDs.Clear();
    }
}
