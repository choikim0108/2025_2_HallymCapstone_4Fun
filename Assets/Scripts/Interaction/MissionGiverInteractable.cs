using UnityEngine;
using Photon.Pun;
using Interaction;

public class MissionGiverInteractable : Interactable
{
    public Transform boxSpawnPoint; // MainScene/Box_SpawnPoint 오브젝트를 Inspector에서 연결
    public string boxesResourcePath = "Boxes"; // Resources/Boxes 폴더

    public override void Interact(PlayerInteraction player)
    {
        // 1) 랜덤 박스 이름 선택
        string[] boxNames = { "Box_Normal", "Box_Fragile", "Box_Heavy", "Box_Explosive", "Box_Frozen" };
        string selectedBoxName = boxNames[Random.Range(0, boxNames.Length)];

        // 2) Resources에서 프리팹 로드
        GameObject boxPrefab = Resources.Load<GameObject>($"{boxesResourcePath}/{selectedBoxName}");
        if (boxPrefab == null)
        {
            Debug.LogError($"[MissionGiverInteractable] 박스 프리팹을 찾을 수 없습니다: {selectedBoxName}");
            return;
        }

        // 3) PhotonNetwork를 통해 박스 생성 (네트워크 동기화)
        if (boxSpawnPoint == null)
        {
            Debug.LogError("[MissionGiverInteractable] boxSpawnPoint가 연결되어 있지 않습니다.");
            return;
        }
        GameObject boxObj = PhotonNetwork.Instantiate($"{boxesResourcePath}/{selectedBoxName}", boxSpawnPoint.position, boxSpawnPoint.rotation);
        // 이름에서 (Clone) 제거
        if (boxObj != null)
        {
            boxObj.name = selectedBoxName;
        }
    }
}
