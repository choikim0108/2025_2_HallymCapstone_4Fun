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
        // Interact UI Prefab 인스턴스도 함께 삭제
        InteractUIHide();

        // 1) EquippedBoxUI 활성화 및 타입별 이미지/텍스트 변경
        Debug.Log($"[BoxInteractable] Interact 호출됨, boxType: {boxType}");
        // 플레이어의 equippedBoxUI가 null이면 자동 할당
        if (player != null && player.equippedBoxUI == null)
        {
            var canvas = GameObject.Find("Canvas");
            if (canvas != null)
            {
                var ui = canvas.transform.Find("EquippedBoxUI");
                if (ui != null)
                    player.equippedBoxUI = ui.gameObject;
            }
        }
        // 이후 equippedBoxUI로 UI 제어
        var uiObj = player != null ? player.equippedBoxUI : equippedBoxUI;
        if (uiObj != null)
        {
            uiObj.SetActive(true);
            // FrozenTimer 활성화 조건
            var frozenTimerObj = uiObj.transform.Find("FrozenTimer");
            if (frozenTimerObj != null)
                frozenTimerObj.gameObject.SetActive(boxType == BoxType.Frozen);
            // ExplosionRateText 활성화 조건
            var explosionRateObj = uiObj.transform.Find("ExplosionRate");
            if (explosionRateObj != null)
            {
                explosionRateObj.gameObject.SetActive(boxType == BoxType.Explosive);
                Debug.Log($"[BoxInteractable] ExplosionRateText 활성화: {boxType == BoxType.Explosive}");
            }
            else
            {
                Debug.LogWarning("[BoxInteractable] explosionRateText 연결 실패");
            }

            // BoxTypeImage 변경
            var typeImageObj = uiObj.transform.Find("BoxTypeImage");
            if (typeImageObj != null)
            {
                var image = typeImageObj.GetComponent<Image>();
                if (image != null)
                {
                    string imgName = boxType.ToString().ToLower();
                    Debug.Log($"[BoxInteractable] 이미지 이름: {imgName}");
                    var sprite = Resources.Load<Sprite>($"Images/{imgName}");
                    Debug.Log($"[BoxInteractable] Sprite 로드: Images/{imgName}, 성공: {sprite != null}");
                    if (sprite != null)
                        image.sprite = sprite;
                    else
                        Debug.LogWarning($"[BoxInteractable] 이미지 로드 실패: Images/{imgName}");
                }
                else
                {
                    Debug.LogWarning("[BoxInteractable] BoxTypeImage에 Image 컴포넌트 없음");
                }
            }
            else
            {
                Debug.LogWarning("[BoxInteractable] EquippedBoxUI에 BoxTypeImage 오브젝트 없음");
            }

            // BoxTypeText 변경
            var typeTextObj = uiObj.transform.Find("BoxTypeText");
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
        else
            Debug.LogError("[BoxInteractable] EquippedBoxUI를 찾을 수 없습니다.");

        // 2) 플레이어가 박스를 "가지고 있음" 상태로 변경 (이름만 저장)
        if (player != null)
        {
            // 플레이어의 equippedTarget에 박스 이름만 저장
            player.equippedTarget = gameObject.name;
            Debug.Log($"[BoxInteractable] Player has equipped {gameObject.name}");
            // 박스 오브젝트는 모든 클라이언트에서 삭제
            if (photonView != null)
            {
                photonView.RPC("DestroyBoxRPC", RpcTarget.All);
                Debug.Log($"[BoxInteractable] {gameObject.name} destroyed on all clients.");
            }
            else
                Debug.LogError($"[BoxInteractable] {gameObject.name} 에 PhotonView가 없습니다.");

            // Explosive/Frozen 특수 로직 시작
            if (boxType == BoxType.Explosive)
            {
                isEquipped = true;
                explosiveTick = 0f;
                Debug.Log($"[BoxInteractable] Explosive 특수 로직 시작! isEquipped: {isEquipped}, explosiveTick: {explosiveTick}");
            }
            if (boxType == BoxType.Frozen)
            {
                frozenActive = true;
                frozenTimer = 0f;
                frozenTargetTime = UnityEngine.Random.Range(180f, 300f); // 3~5분
                if (frozenTimerText != null)
                    frozenTimerText.gameObject.SetActive(true);
                Debug.Log($"[BoxInteractable] Frozen 특수 로직 시작! frozenActive: {frozenActive}, frozenTimer: {frozenTimer}, frozenTargetTime: {frozenTargetTime}");
            }
        }
    }
    // Explosive 박스용: 들고 있는 시간, 폭발 확률
    private float explosiveTick = 0f;
    private float explosiveBaseChance = 0.01f; // 시작 확률 1%
    private float explosiveChanceIncrease = 0.01f; // 초당 1% 증가
    private bool isEquipped = false;

    private TMPro.TextMeshProUGUI explosionRateText;

    // Frozen 박스용: 타이머
    private float frozenTimer = 0f;
    private float frozenTargetTime = 0f;
    private bool frozenActive = false;
    private TMPro.TextMeshProUGUI frozenTimerText;

    private MissionStart missionStart;
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

        // MissionStart 연결
        var msObj = GameObject.FindFirstObjectByType<MissionStart>();
        if (msObj != null)
            missionStart = msObj;

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
        var playerObj = GameObject.FindWithTag("Player");
        if (playerObj != null)
        {
            var playerInteraction = playerObj.GetComponent<PlayerInteraction>();
            var playerPhotonView = playerObj.GetComponent<PhotonView>();
            // Explosive 박스: tick마다 확률 증가, 폭발 처리
            if (playerInteraction.equippedTarget != null && playerInteraction.equippedTarget.Contains("Explosive"))
            {
                explosiveTick += Time.deltaTime;
                float chance = explosiveBaseChance + explosiveChanceIncrease * explosiveTick;
                if (explosionRateText != null)
                    explosionRateText.text = $"{Mathf.RoundToInt(chance * 100f)}%";
                if (UnityEngine.Random.value < chance * Time.deltaTime)
                {
                    photonView.RPC("ExplodeBoxRPC", RpcTarget.All);
                    isEquipped = false;
                    explosiveTick = 0f;
                    if (playerPhotonView != null)
                        photonView.RPC("MissionFailRPC", RpcTarget.All, playerPhotonView.ViewID, "Explosive 박스 폭발");
                }
            }

            // Frozen 박스: 타이머 갱신 및 시간 종료 시 파괴
            if (playerInteraction.equippedTarget != null && playerInteraction.equippedTarget.Contains("Frozen"))
            {
                frozenTimer += Time.deltaTime;
                if (frozenTimerText != null)
                {
                    int remain = Mathf.CeilToInt(frozenTargetTime - frozenTimer);
                    frozenTimerText.text = $"{remain / 60:D2}:{remain % 60:D2}";
                }
                if (frozenTimer >= frozenTargetTime)
                {
                    if (playerPhotonView != null)
                        photonView.RPC("MissionFailRPC", RpcTarget.All, playerPhotonView.ViewID, "Frozen 박스 시간 초과");
                    frozenActive = false;
                    if (frozenTimerText != null)
                        frozenTimerText.gameObject.SetActive(false);
                }
            }
        }
    }

    [PunRPC]
    public void MissionFailRPC(int playerViewID, string reason)
    {
        // 해당 ViewID의 플레이어만 실패 처리
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
        if (missionStart != null)
            missionStart.MissionFail(reason);
        // 박스 오브젝트 삭제는 DestroyBoxRPC에서만 수행
    }

    [PunRPC]
    public void DestroyBoxRPC()
    {
        Destroy(gameObject);
    }

    [PunRPC]
    public void ExplodeBoxRPC()
    {
        // 현재 플레이어 찾기
        var player = GameObject.FindWithTag("Player");
        if (player != null)
        {
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
