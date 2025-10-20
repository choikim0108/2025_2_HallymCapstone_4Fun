using UnityEngine;
using Interaction;
using UnityEngine.UI;
using Photon.Pun;
using Controller;

public enum BoxType
{
    Normal,
    Fragile,
    Heavy, // 이동
    Explosive, // 충격 시 폭발
    Frozen     // 일정 시간 후 녹음
}

public class BoxInteractable : Interactable
{
    // 추상 메서드 구현: 플레이어 정보를 받는 상호작용
    public override void Interact(PlayerInteraction player)
    {
        if (player == null)
        {
            InteractUIHide();
            return;
        }

        if (!player.CanEquipNewTarget(gameObject.name))
            return;

        // Interact UI Prefab 인스턴스도 함께 삭제
        InteractUIHide();

        // 1) EquippedBoxUI 활성화 및 타입별 이미지/텍스트 변경
    //Debug.Log($"[BoxInteractable][Interact] boxType: {boxType}, player: {player?.name}, playerViewID: {player?.GetComponent<PhotonView>()?.ViewID}");
        // 플레이어의 equippedBoxUI가 null이면 자동 할당
        if (player.equippedBoxUI == null)
        {
            var canvas = GameObject.Find("Canvas");
            if (canvas != null)
            {
                var ui = canvas.transform.Find("EquippedBoxUI");
                if (ui != null)
                    player.equippedBoxUI = ui.gameObject;
                //Debug.Log($"[BoxInteractable][Interact] EquippedBoxUI 자동 할당: {player.equippedBoxUI != null}");
            }
        }
        // 박스 equip 직후 UI 컴포넌트 연결 재시도
        player.ConnectEquippedBoxUI();
        // 이후 equippedBoxUI로 UI 제어
        var uiObj = player.equippedBoxUI ?? equippedBoxUI;
            //Debug.Log($"[BoxInteractable][Interact] EquippedBoxUI 찾기: {(uiObj != null ? "성공" : "실패")}, player: {player?.name}");
        if (uiObj != null)
        {
            uiObj.SetActive(true);
                        //Debug.Log($"[BoxInteractable][Interact] EquippedBoxUI 활성화: {uiObj.activeSelf}");
            // EquippedBoxPanel 아래에서 특수 텍스트 제어 (활성화만, 텍스트 값은 PlayerInteraction에서 갱신)
            var panel = uiObj.transform.Find("EquippedBoxPanel");
            if (panel != null)
            {
                // FrozenTimer 활성화 조건
                var frozenTimerObj = panel.Find("FrozenTimer");
                if (frozenTimerObj != null)
                    frozenTimerObj.gameObject.SetActive(boxType == BoxType.Frozen);
                // ExplosionRateText 활성화 조건
                var explosionRateObj = panel.Find("ExplosionRate");
                if (explosionRateObj != null)
                    explosionRateObj.gameObject.SetActive(boxType == BoxType.Explosive);
                // BoxTypeImage 변경
                var typeImageObj = panel.Find("BoxTypeImage");
                if (typeImageObj != null)
                {
                    var image = typeImageObj.GetComponent<Image>();
                    if (image != null)
                    {
                        string imgName = boxType.ToString().ToLower();
                        var sprite = Resources.Load<Sprite>($"Images/{imgName}");
                        if (sprite != null)
                            image.sprite = sprite;
                    }
                }
                // BoxTypeText 변경
                var typeTextObj = panel.Find("BoxTypeText");
                if (typeTextObj != null)
                {
                    var tmp = typeTextObj.GetComponent<TMPro.TextMeshProUGUI>();
                    if (tmp != null)
                    {
                        string typeText = "";
                        switch (boxType)
                        {
                            case BoxType.Normal: typeText = "Normal Box"; break;
                            case BoxType.Fragile: typeText = "Fragile Box"; break;
                            case BoxType.Heavy: typeText = "Heavy Box"; break;
                            case BoxType.Explosive: typeText = "Explosive Box"; break;
                            case BoxType.Frozen: typeText = "Frozen Box"; break;
                        }
                        tmp.text = typeText;
                    }
                }
            }
        }
        else
            Debug.LogError($"[BoxInteractable][Interact] EquippedBoxUI를 찾을 수 없습니다. player: {player?.name}");

        // 2) 플레이어가 박스를 "가지고 있음" 상태로 변경 (데이터 전달)
        player.equippedTarget = gameObject.name;
        player.equippedBoxType = boxType;
        if (boxType == BoxType.Explosive)
        {
            player.explosiveTick = 0f;
            player.explosiveBaseChance = explosiveBaseChance;
            player.explosiveChanceIncrease = explosiveChanceIncrease;
            player.isExplosiveEquipped = true;
        }
        else
        {
            player.isExplosiveEquipped = false;
        }
        if (boxType == BoxType.Frozen)
        {
            player.frozenTimer = 0f;
            player.frozenTargetTime = UnityEngine.Random.Range(180f, 300f);
            player.isFrozenEquipped = true;
        }
        else
        {
            player.isFrozenEquipped = false;
        }
        //Debug.Log($"[BoxInteractable][Interact] PlayerInteraction에 박스 데이터 전달: type={boxType}, explosive={player.isExplosiveEquipped}, frozen={player.isFrozenEquipped}");
        if (photonView != null)
        {
            photonView.RPC("DestroyBoxRPC", RpcTarget.All);
        }
    }
    // Explosive 박스용: 들고 있는 시간, 폭발 확률
    private float explosiveTick = 0f;
    private float explosiveBaseChance = 0.01f; // 시작 확률 1%
    private float explosiveChanceIncrease = 0.0f; // 0.01f, 테스트용으로 0으로 수정해둠. 초당 1% 증가
    private bool isEquipped = false;
    private int equippedPlayerViewID = -1; // 박스를 든 플레이어의 ViewID

    private TMPro.TextMeshProUGUI explosionRateText;

    // Frozen 박스용: 타이머
    private float frozenTimer = 0f;
    private float frozenTargetTime = 0f;
    private bool frozenActive = false;
    private TMPro.TextMeshProUGUI frozenTimerText;
    private int frozenPlayerViewID = -1; // Frozen 박스 플레이어 ViewID (동일하게 사용)

    private MissionManager missionManager;
    public BoxType boxType;
    private PhotonView photonView;

    // Inspector에서 직접 연결할 수 있도록 public 필드 추가
    public GameObject equippedBoxUI;

    private void Awake()
    {
        photonView = GetComponent<PhotonView>();
        // 만약 Inspector에서 연결하지 않았다면, 계층에서 찾아서 할당 (비활성화 포함)
        if (equippedBoxUI == null)
        {
            var canvas = GameObject.Find("Canvas");
            if (canvas != null)
            {
                var ui = canvas.transform.Find("EquippedBoxUI");
                if (ui != null)
                    equippedBoxUI = ui.gameObject;
            }
        }

        // MissionManager 연결
        var msObj = MissionManager.Instance ?? GameObject.FindFirstObjectByType<MissionManager>();
        if (msObj != null)
            missionManager = msObj;

        // Frozen 타이머 텍스트 연결
        if (equippedBoxUI != null)
        {
            var frozenTimerObj = equippedBoxUI.transform.Find("FrozenTimer");
            if (frozenTimerObj != null)
                frozenTimerText = frozenTimerObj.GetComponent<TMPro.TextMeshProUGUI>();
            // ExplosionRateText 연결
            var explosionRateObj = equippedBoxUI.transform.Find("ExplosionRate");
            if (explosionRateObj != null)
            {
                explosionRateText = explosionRateObj.GetComponent<TMPro.TextMeshProUGUI>();
                Debug.Log($"[BoxInteractable] ExplosionRateText 연결됨: {explosionRateText != null}");
            }
            else
            {
                Debug.LogWarning("[BoxInteractable] EquippedBoxUI에 ExplosionRate 오브젝트 없음");
            }
        }
    }

    // 박스 특성별 동작
    public void OnPlayerCollision(float collisionSpeed)
    {
        CheckBoxCondition(collisionSpeed);
    }

    // 박스 조건 검사 및 실패 처리
    public void CheckBoxCondition(float collisionSpeed)
    {
        var player = GameObject.FindWithTag("Player");
        if (player == null) return;
        var playerInteraction = player.GetComponent<PlayerInteraction>();
        if (playerInteraction == null) return;
        var playerPhotonView = player.GetComponent<PhotonView>();
        if (playerPhotonView == null) return;

        // 플레이어가 이 박스를 운송 중인지 확인
        if (playerInteraction.equippedTarget != gameObject.name) return;

        switch (boxType)
        {
            case BoxType.Fragile:
                if (collisionSpeed > 5f)
                {
                    photonView.RPC("MissionFailRPC", RpcTarget.All, playerPhotonView.ViewID, "Fragile 박스 파손");
                }
                break;
            case BoxType.Heavy:
                break;
            case BoxType.Explosive:
                if (collisionSpeed > 7f)
                {
                    photonView.RPC("MissionFailRPC", RpcTarget.All, playerPhotonView.ViewID, "Explosive 박스 폭발");
                }
                break;
            case BoxType.Frozen:
                break;
        }
    }
    private void Update()
    {
        // 박스 특수 로직은 PlayerInteraction에서만 관리. 박스는 데이터만 넘김.
    }

    [PunRPC]
    public void MissionFailRPC(int playerViewID, string reason)
    {
        // 해당 ViewID의 플레이어만 실패 처리
        Debug.Log($"[BoxInteractable][RPC] MissionFailRPC 호출됨, playerViewID={playerViewID}, reason={reason}");
        PhotonView targetView = PhotonView.Find(playerViewID);
        if (targetView != null)
        {
            var player = targetView.gameObject;
            var playerInteraction = player.GetComponent<PlayerInteraction>();
            if (playerInteraction != null)
            {
                playerInteraction.equippedTarget = null;
                // EquippedBoxUI와 모든 자식 비활성화
                if (playerInteraction.equippedBoxUI != null)
                {
                    playerInteraction.equippedBoxUI.SetActive(false);
                    foreach (Transform child in playerInteraction.equippedBoxUI.transform)
                        child.gameObject.SetActive(false);
                }
            }
        }
        if (missionManager != null)
            missionManager.MissionFail(reason);
        // 박스 오브젝트 삭제는 DestroyBoxRPC에서만 수행
    }

    [PunRPC]
    public void DestroyBoxRPC()
    {
        Destroy(gameObject);
    }

    [PunRPC]
    public void ExplodeBoxRPC(int playerViewID)
    {
        PhotonView targetView = PhotonView.Find(playerViewID);
        Debug.Log($"[ExplodeBox][RPC] ExplodeBoxRPC 호출됨, playerViewID={playerViewID}");
        if (targetView != null)
        {
            var player = targetView.gameObject;
            // 폭발 사운드 재생
            var explosionClip = Resources.Load<AudioClip>("Sounds/explosion");
            if (explosionClip != null)
                AudioSource.PlayClipAtPoint(explosionClip, player.transform.position);
            else
                Debug.LogWarning("[BoxInteractable] 폭발 사운드 로드 실패: Sounds/explosion");

            // 폭발 이펙트 생성 및 자동 삭제
            var explosionPrefab = Resources.Load<GameObject>("Effects/BigExplosion");
            if (explosionPrefab != null)
            {
                var effect = GameObject.Instantiate(explosionPrefab, player.transform.position, Quaternion.identity);
                var ps = effect.GetComponent<ParticleSystem>();
                if (ps != null)
                {
                    ps.Play();
                    GameObject.Destroy(effect, ps.main.duration + 0.5f);
                }
                else
                {
                    GameObject.Destroy(effect, 3f); // ParticleSystem 없으면 3초 후 삭제
                }
            }
            else
                Debug.LogWarning("[BoxInteractable] 폭발 이펙트 로드 실패: Effects/BigExplosion");

            // Rigidbody force 적용
            var rb = player.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.AddForce(Vector3.up * 1000f, ForceMode.Impulse); // 하늘로 날리기
            }
        }
    }

    [PunRPC]
    public void SetBoxActiveRPC(bool isActive)
    {
        gameObject.SetActive(isActive);
    }
}
