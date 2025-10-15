using UnityEngine;
using Photon.Pun;
using Interaction;
using System;
using UnityEngine.UI;

public class PlayerInteraction : MonoBehaviourPun
{
    // 박스 equip 시 UI 컴포넌트 연결을 동적으로 수행
    public void ConnectEquippedBoxUI()
    {
        Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] equippedBoxUI 연결 시도: {equippedBoxUI != null}");
        explosionRateText = null;
        frozenTimerText = null;
        boxTypeImage = null;
        boxTypeText = null;
        if (equippedBoxUI != null)
        {
            var panel = equippedBoxUI.transform.Find("EquippedBoxPanel");
            if (panel != null)
            {
                var explosionRateObj = panel.Find("ExplosionRate");
                if (explosionRateObj != null)
                    explosionRateText = explosionRateObj.GetComponent<TMPro.TextMeshProUGUI>();
                var frozenTimerObj = panel.Find("FrozenTimer");
                if (frozenTimerObj != null)
                    frozenTimerText = frozenTimerObj.GetComponent<TMPro.TextMeshProUGUI>();
                var typeImageObj = panel.Find("BoxTypeImage");
                if (typeImageObj != null)
                    boxTypeImage = typeImageObj.GetComponent<Image>();
                var typeTextObj = panel.Find("BoxTypeText");
                if (typeTextObj != null)
                    boxTypeText = typeTextObj.GetComponent<TMPro.TextMeshProUGUI>();
            }
        }
        Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] ExplosionRateText 연결: {explosionRateText != null}");
        Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] FrozenTimerText 연결: {frozenTimerText != null}");
        Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] BoxTypeImage 연결: {boxTypeImage != null}");
        Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] BoxTypeText 연결: {boxTypeText != null}");
    }
    // 박스 특수 로직 관리용 변수
    public BoxType equippedBoxType;
    public bool isExplosiveEquipped = false;
    public float explosiveTick = 0f;
    public float explosiveBaseChance = 0.01f;
    public float explosiveChanceIncrease = 0.01f;

    public bool isFrozenEquipped = false;
    public float frozenTimer = 0f;
    public float frozenTargetTime = 0f;

    // UI 관련 참조
    private TMPro.TextMeshProUGUI explosionRateText;
    private TMPro.TextMeshProUGUI frozenTimerText;
    private Image boxTypeImage;
    private TMPro.TextMeshProUGUI boxTypeText;

    public float checkRadius = 2f; // 상호작용 감지 반경
    public LayerMask interactableLayer; // Interactable 오브젝트만 감지
    public Interactable currentTarget;
    public string equippedTarget;
    public GameObject equippedBoxUI; // BoxInteractable에서 직접 제어할 수 있도록 참조 추가

    void Awake()
    {
        // Canvas에서 EquippedBoxUI 찾아서 할당
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
        // 최초 비활성화
        if (equippedBoxUI != null)
            equippedBoxUI.SetActive(false);
    }

    void Start()
    {

    }

    void Update()
    {
        if (!photonView.IsMine) return; // 내 플레이어만 처리
        CheckInteractable();
        HandleInteractionInput();

        // equippedTarget(string)으로 박스 상태 일괄 관리
        if (string.IsNullOrEmpty(equippedTarget))
        {
            // 박스가 없으면 모든 특수 로직/변수/텍스트 초기화
            isExplosiveEquipped = false;
            isFrozenEquipped = false;
            explosiveTick = 0f;
            frozenTimer = 0f;
            if (explosionRateText != null)
                explosionRateText.gameObject.SetActive(false);
            if (frozenTimerText != null)
                frozenTimerText.gameObject.SetActive(false);
            if (equippedBoxUI != null)
                equippedBoxUI.SetActive(false);
            return;
        }

        // 박스 특수 로직 관리
        // 박스 equip 시 값 초기화 보장
        if (isExplosiveEquipped && explosiveTick == 0f)
        {
            explosiveTick = 0f;
        }
        if (isFrozenEquipped && frozenTimer == 0f)
        {
            frozenTimer = 0f;
        }

        // 폭발 박스 UI 실시간 갱신
        if (isExplosiveEquipped)
        {
            explosiveTick += Time.deltaTime;
            float chance = explosiveBaseChance + explosiveChanceIncrease * explosiveTick;
            if (explosionRateText != null && explosionRateText.gameObject.activeInHierarchy)
            {
                explosionRateText.text = $"{Mathf.RoundToInt(chance * 100f)}%";
            }
            if (UnityEngine.Random.value < chance * Time.deltaTime)
            {
                Debug.Log($"[PlayerInteraction][Explosive] 폭발 발생! chance={chance}");
                photonView.RPC("ExplodeBoxRPC", RpcTarget.All, photonView.ViewID);
                photonView.RPC("MissionFailRPC", RpcTarget.All, photonView.ViewID, "Explosive 박스 폭발");
                isExplosiveEquipped = false;
                explosiveTick = 0f;
            }
        }
        else if (explosionRateText != null)
        {
            explosionRateText.gameObject.SetActive(false);
        }
        // Frozen 박스 UI 실시간 갱신
        if (isFrozenEquipped)
        {
            frozenTimer += Time.deltaTime;
            if (frozenTimerText != null && frozenTimerText.gameObject.activeInHierarchy)
            {
                int remain = Mathf.CeilToInt(frozenTargetTime - frozenTimer);
                frozenTimerText.text = $"{remain / 60:D2}:{remain % 60:D2}";
            }
            if (frozenTimer >= frozenTargetTime)
            {
                Debug.Log($"[PlayerInteraction][Frozen] 시간 초과!");
                photonView.RPC("MissionFailRPC", RpcTarget.All, photonView.ViewID, "Frozen 박스 시간 초과");
                isFrozenEquipped = false;
                frozenTimer = 0f;
            }
        }
        else if (frozenTimerText != null)
        {
            frozenTimerText.gameObject.SetActive(false);
        }
        // UI BoxTypeImage, BoxTypeText 갱신
        if (equippedBoxUI != null && equippedTarget != null)
        {
            equippedBoxUI.SetActive(true);
            if (boxTypeImage != null)
            {
                string imgName = equippedBoxType.ToString().ToLower();
                var sprite = Resources.Load<Sprite>($"Images/{imgName}");
                boxTypeImage.sprite = sprite;
            }
            if (boxTypeText != null)
            {
                string typeText = "";
                switch (equippedBoxType)
                {
                    case BoxType.Normal: typeText = "Normal Box"; break;
                    case BoxType.Fragile: typeText = "Fragile Box"; break;
                    case BoxType.Heavy: typeText = "Heavy Box"; break;
                    case BoxType.Explosive: typeText = "Explosive Box"; break;
                    case BoxType.Frozen: typeText = "Frozen Box"; break;
                }
                boxTypeText.text = typeText;
            }
        }
    }

    void CheckInteractable()
    {
        Collider[] hits = Physics.OverlapSphere(transform.position, checkRadius, interactableLayer);
        Interactable nearest = null;
        float minDist = float.MaxValue;
        foreach (var hit in hits)
        {
            var interactable = hit.GetComponent<Interactable>();
            if (interactable != null)
            {
                float dist = Vector3.Distance(transform.position, interactable.transform.position);
                if (dist < interactable.interactDistance && dist < minDist)
                {
                    nearest = interactable;
                    minDist = dist;
                }
            }
        }
        if (nearest != null)
        {
            if (currentTarget != nearest)
            {
                if (currentTarget != null) currentTarget.InteractUIHide();
                currentTarget = nearest;
                currentTarget.InteractUIShow(currentTarget.gameObject.name, currentTarget.transform);
                var nearestPhotonView = nearest.GetComponent<PhotonView>();
                if (nearestPhotonView != null)
                {
                    photonView.RPC("SetCurrentInteractableRPC", RpcTarget.All, nearestPhotonView.ViewID);
                }
                else
                {
                    photonView.RPC("SetCurrentInteractableRPC", RpcTarget.All, -1);
                }
            }
        }
        else
        {
            if (currentTarget != null)
            {
                currentTarget.InteractUIHide();
                currentTarget = null;
                photonView.RPC("SetCurrentInteractableRPC", RpcTarget.All, -1);
            }
        }
    }

    void HandleInteractionInput()
    {
        if (currentTarget != null && Input.GetKeyDown(KeyCode.F))
        {
            currentTarget.Interact(this);
            Debug.Log($"[PlayerInteraction] Interacted with {currentTarget.gameObject.name}");
        }
    }

    [PunRPC]
    public void SetCurrentInteractableRPC(int viewID)
    {
        if (viewID == -1)
        {
            currentTarget = null;
        }
        else
        {
            var obj = PhotonView.Find(viewID);
            if (obj != null)
            {
                currentTarget = obj.GetComponent<Interactable>();
            }
        }
    }

    [PunRPC]
    public void ExplodeBoxRPC(int playerViewID)
    {
        Debug.Log($"[PlayerInteraction][RPC] ExplodeBoxRPC 호출됨, playerViewID={playerViewID}, 내 ViewID={photonView.ViewID}");
        if (photonView.ViewID != playerViewID) return;
        // 폭발 사운드 재생
        var explosionClip = Resources.Load<AudioClip>("Sounds/explosion");
        if (explosionClip != null)
            AudioSource.PlayClipAtPoint(explosionClip, transform.position);
        else
            Debug.LogWarning("[PlayerInteraction][RPC] 폭발 사운드 로드 실패: Sounds/explosion");
        // 폭발 이펙트 생성 및 자동 삭제
        var explosionPrefab = Resources.Load<GameObject>("Effects/BigExplosion");
        if (explosionPrefab != null)
        {
            var effect = GameObject.Instantiate(explosionPrefab, transform.position, Quaternion.identity);
            var ps = effect.GetComponent<ParticleSystem>();
            if (ps != null)
            {
                ps.Play();
                GameObject.Destroy(effect, ps.main.duration + 0.5f);
            }
            else
            {
                GameObject.Destroy(effect, 3f);
            }
        }
        else
            Debug.LogWarning("[PlayerInteraction][RPC] 폭발 이펙트 로드 실패: Effects/BigExplosion");
        // Rigidbody force 적용
        var rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.AddForce(Vector3.up * 1000f, ForceMode.Impulse);
        }
    }

    [PunRPC]
    public void MissionFailRPC(int playerViewID, string reason)
    {
        Debug.Log($"[PlayerInteraction][RPC] MissionFailRPC 호출됨, playerViewID={playerViewID}, 내 ViewID={photonView.ViewID}, reason={reason}");
        if (photonView.ViewID != playerViewID) return;
        equippedTarget = null;
        // EquippedBoxUI만 비활성화 (내부 자식 오브젝트는 그대로 둠)
        if (equippedBoxUI != null)
        {
            equippedBoxUI.SetActive(false);
        }
        isExplosiveEquipped = false;
        isFrozenEquipped = false;
        // 추가 실패 처리 필요시 여기에 작성
    }
}