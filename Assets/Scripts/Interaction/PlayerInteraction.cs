using UnityEngine;
using Photon.Pun;
using Interaction;
using System;
using UnityEngine.UI;
using Controller; // MovePlayerInput 클래스를 사용하기 위해 추가

public class PlayerInteraction : MonoBehaviourPun
{
    // --- 기존 변수 선언 (변경 없음) ---
    public BoxType equippedBoxType;
    public bool isExplosiveEquipped = false;
    public float explosiveTick = 0f;
    public float explosiveBaseChance = 0.01f;
    public float explosiveChanceIncrease = 0.01f;
    public bool isFrozenEquipped = false;
    public float frozenTimer = 0f;
    public float frozenTargetTime = 0f;
    private TMPro.TextMeshProUGUI explosionRateText;
    private TMPro.TextMeshProUGUI frozenTimerText;
    private Image boxTypeImage;
    private TMPro.TextMeshProUGUI boxTypeText;
    public float checkRadius = 2f;
    public LayerMask interactableLayer;
    public Interactable currentTarget;
    public bool debugInteractable = false; // 디버그 로그 토글
    public string equippedTarget;
    public bool isInDialogue = false; // 대화 중인지 여부
    public GameObject equippedBoxUI;

    // --- 버스 탑승 관련 변수 추가 ---
    private bool isRidingBus = false;       // 버스에 탑승 중인지 상태를 저장하는 변수
    private GameObject currentBus = null;   // 현재 탑승한 버스 오브젝트
    private CharacterMover characterMover;  // 플레이어 이동 로직을 제어하기 위한 참조
    private MovePlayerInput movePlayerInput; // 플레이어 입력을 제어하기 위한 참조
    private CharacterController characterController; // 플레이어의 물리적 충돌을 제어하기 위한 참조
    private Rigidbody playerRigidbody;

    void Awake()
    {
        // --- 기존 Awake 코드 (변경 없음) ---
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
        if (equippedBoxUI != null)
            equippedBoxUI.SetActive(false);
        
        // --- 버스 탑승 관련 컴포넌트 참조 추가 ---
        // 플레이어 이동을 비활성화/활성화하기 위해 컴포넌트들을 미리 찾아둡니다.
        characterMover = GetComponent<CharacterMover>();
        movePlayerInput = GetComponent<MovePlayerInput>();
        characterController = GetComponent<CharacterController>();
        playerRigidbody = GetComponent<Rigidbody>();
    }

    void Update()
    {
        if (!photonView.IsMine) return;

        HandleInteractionInput(); // 상호작용 입력을 항상 체크합니다.

        // 버스에 타지 않았을 때만 주변 상호작용 오브젝트를 탐색합니다.
        if (!isRidingBus)
        {
            CheckInteractable();
        }

        // --- 기존 박스 관련 Update 로직 (변경 없음) ---
        if (string.IsNullOrEmpty(equippedTarget))
        {
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
        if (isExplosiveEquipped && explosiveTick == 0f)
        {
            explosiveTick = 0f;
        }
        if (isFrozenEquipped && frozenTimer == 0f)
        {
            frozenTimer = 0f;
        }
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
                //Debug.Log($"[PlayerInteraction][Explosive] 폭발 발생! chance={chance}");
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
                //Debug.Log($"[PlayerInteraction][Frozen] 시간 초과!");
                photonView.RPC("MissionFailRPC", RpcTarget.All, photonView.ViewID, "Frozen 박스 시간 초과");
                isFrozenEquipped = false;
                frozenTimer = 0f;
            }
        }
        else if (frozenTimerText != null)
        {
            frozenTimerText.gameObject.SetActive(false);
        }
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
    
    // --- 상호작용 입력 처리 로직 수정 ---
    void HandleInteractionInput()
    {
        if (Input.GetKeyDown(KeyCode.F))
        {
            if (isRidingBus)
            {
                // 버스에 타고 있다면, 하차를 시도합니다.
                AttemptToGetOffBus();
            }
            else if (currentTarget != null)
            {
                // 버스에 타지 않았고 상호작용 대상이 있다면, 그 대상과 상호작용합니다.
                currentTarget.Interact(this);
                //Debug.Log($"[PlayerInteraction] Interacted with {currentTarget.gameObject.name}");
            }
        }
    }

    public bool CanEquipNewTarget(string attemptedTargetName)
    {
        if (string.IsNullOrEmpty(equippedTarget))
            return true;

        NotifyEquipBlocked(attemptedTargetName);
        return false;
    }

    public void NotifyEquipBlocked(string attemptedTargetName)
    {
        if (!photonView.IsMine)
            return;

        string currentName = string.IsNullOrEmpty(equippedTarget) ? "Unknown" : equippedTarget;
        string attemptedName = string.IsNullOrEmpty(attemptedTargetName) ? "Unknown" : attemptedTargetName;
        string message = $"You can't equip \"{attemptedName}\" because you already equipped \"{currentName}\"";
        Debug.LogWarning($"[PlayerInteraction] {message}");
        // TODO: integrate with a player-facing UI notification channel.
    }

    // --- 버스 탑승/하차 관련 메서드 추가 ---

    /// BusInteractable에 의해 호출되어 버스 탑승을 시도합니다.
    public void AttemptToBoardBus(GameObject bus)
    {
        // 내 플레이어가 아니거나, 이미 버스에 탑승 중이면 아무것도 하지 않습니다.
        if (!photonView.IsMine || isRidingBus) return;

        var busPhotonView = bus.GetComponent<PhotonView>();
        if (busPhotonView != null)
        {
            // 마스터 클라이언트에게 탑승을 요청하는 RPC를 보냅니다.
            busPhotonView.RPC("BoardRequestRPC", RpcTarget.MasterClient, photonView.ViewID);
        }
    }

    /// F키를 눌렀을 때 버스 하차를 시도합니다.
    private void AttemptToGetOffBus()
    {
        // 내 플레이어가 아니거나, 버스에 탑승 중이 아니면 아무것도 하지 않습니다.
        if (!photonView.IsMine || !isRidingBus || currentBus == null) return;

        var busPhotonView = currentBus.GetComponent<PhotonView>();
        if (busPhotonView != null)
        {
            // 마스터 클라이언트에게 하차를 요청하는 RPC를 보냅니다.
            busPhotonView.RPC("GetOffRequestRPC", RpcTarget.MasterClient, photonView.ViewID);
        }
    }
    

    /// 운전자로 버스에 탑승했을 때 BusController에 의해 호출됩니다.
    public void OnBoardedAsDriver(GameObject bus)
    {
        BoardBus(bus);
    }

    /// 승객으로 버스에 탑승했을 때 BusController에 의해 호출됩니다.
    public void OnBoardedAsPassenger(GameObject bus)
    {
        BoardBus(bus);
    }


    /// 버스에 탑승하는 공통 로직을 처리합니다.
    private void BoardBus(GameObject bus)
    {
        isRidingBus = true;
        currentBus = bus;

        // --- [3] Rigidbody의 물리 효과 비활성화 ---
        if (playerRigidbody != null)
        {
            playerRigidbody.useGravity = false; // 중력 끄기
            playerRigidbody.isKinematic = true;   // 물리 효과 끄기
        }

        // 플레이어 이동 관련 컴포넌트들을 비활성화하여 조작을 막습니다.
        //if (movePlayerInput != null) movePlayerInput.enabled = false; // <- 카메라 비활성화 수정 202510181400
        if (characterMover != null) characterMover.enabled = false;
        if (characterController != null) characterController.enabled = false;
        
        // 플레이어 캐릭터가 보이지 않도록 모든 렌더러를 비활성화합니다.
        foreach (var renderer in GetComponentsInChildren<Renderer>(true))
        {
            renderer.enabled = false;
        }

        // 플레이어를 버스의 자식으로 만들어 함께 움직이게 합니다.
        transform.SetParent(bus.transform);
        // 버스 내의 상대적 위치와 회전을 초기화합니다. (필요에 따라 조정 가능)
        transform.localPosition = Vector3.zero;
        transform.localRotation = Quaternion.identity;
    }

    /// 버스에서 하차할 때 BusController에 의해 호출됩니다.
    public void OnGetOffBus(Vector3 exitPosition)
    {
        if (!isRidingBus) return;

        // 버스와의 부모-자식 관계를 해제합니다.
        transform.SetParent(null);
        
        // BusController가 지정한 하차 위치로 플레이어를 이동시킵니다.
        // CharacterController가 비활성화된 상태에서 위치를 먼저 설정해야 순간이동이 원활합니다.
        transform.position = exitPosition;

        // --- [4] Rigidbody 물리 효과 다시 활성화 ---
        // (CharacterController보다 먼저 켜는 것이 안전할 수 있음)
        if (playerRigidbody != null)
        {
            playerRigidbody.useGravity = true; // 중력 다시 켜기
            playerRigidbody.isKinematic = false;  // 물리 효과 다시 켜기
        }

        // 비활성화했던 플레이어 이동 관련 컴포넌트들을 다시 활성화합니다.
        if (characterController != null) characterController.enabled = true;
        //if (movePlayerInput != null) movePlayerInput.enabled = true; // <- 카메라 비활성화 수정 202510181400
        if (characterMover != null) characterMover.enabled = true;
        
        // 플레이어 캐릭터가 다시 보이도록 모든 렌더러를 활성화합니다.
        foreach (var renderer in GetComponentsInChildren<Renderer>(true))
        {
            renderer.enabled = true;
        }

        // 상태 변수를 초기화합니다.
        isRidingBus = false;
        currentBus = null;
    }


    // --- 기존 코드 (변경 없음) ---
    public void ConnectEquippedBoxUI()
    {
    //Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] equippedBoxUI 연결 시도: {equippedBoxUI != null}");
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
    //Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] ExplosionRateText 연결: {explosionRateText != null}");
    //Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] FrozenTimerText 연결: {frozenTimerText != null}");
    //Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] BoxTypeImage 연결: {boxTypeImage != null}");
    //Debug.Log($"[PlayerInteraction][ConnectEquippedBoxUI] BoxTypeText 연결: {boxTypeText != null}");
    }

    /* 수정 20251018 
    void CheckInteractable()
    {
        Collider[] hits = Physics.OverlapSphere(transform.position, checkRadius, interactableLayer);
        Interactable nearest = null;
        float minDist = float.MaxValue;
        if (debugInteractable) Debug.Log($"[PlayerInteraction] OverlapSphere hits: {hits.Length}");
        foreach (var hit in hits)
        {
            if (debugInteractable) Debug.Log($"[PlayerInteraction] Hit collider: {hit.name}");
            // try direct component first, then parent (colliders are often on child objects)
            var interactable = hit.GetComponent<Interactable>();
            if (interactable == null)
                interactable = hit.GetComponentInParent<Interactable>();

            if (interactable != null)
            {
                if (debugInteractable) Debug.Log($"[PlayerInteraction] Found interactable on: {interactable.gameObject.name}");
                float dist = Vector3.Distance(transform.position, interactable.transform.position);
                if (dist < interactable.interactDistance && dist < minDist)
                {
                    nearest = interactable;
                    minDist = dist;
                }
            }
            else
            {
                if (debugInteractable) Debug.Log($"[PlayerInteraction] No Interactable component on {hit.name} or its parents.");
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
    */
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
                currentTarget = nearest; // 로컬 변수 설정
                currentTarget.InteractUIShow(currentTarget.gameObject.name, currentTarget.transform);

            }
        }
        else
        {
            if (currentTarget != null)
            {
                currentTarget.InteractUIHide();
                currentTarget = null;
            }
        }
    }
    /* 비활성화 수정 20251018 
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
    */

    [PunRPC]
    public void ExplodeBoxRPC(int playerViewID)
    {
        Debug.Log($"[PlayerInteraction][RPC] ExplodeBoxRPC 호출됨, playerViewID={playerViewID}, 내 ViewID={photonView.ViewID}");
        if (photonView.ViewID != playerViewID) return;
        var explosionClip = Resources.Load<AudioClip>("Sounds/explosion");
        if (explosionClip != null)
            AudioSource.PlayClipAtPoint(explosionClip, transform.position);
        else
            Debug.LogWarning("[PlayerInteraction][RPC] 폭발 사운드 로드 실패: Sounds/explosion");
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
        if (equippedBoxUI != null)
        {
            equippedBoxUI.SetActive(false);
        }
        isExplosiveEquipped = false;
        isFrozenEquipped = false;
    }
}