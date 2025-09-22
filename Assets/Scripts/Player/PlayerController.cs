
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;  // New Input System을 사용하기 위해 필요
using Interaction;

[RequireComponent(typeof(Player))]
[RequireComponent(typeof(Rigidbody))]
public class PlayerController : MonoBehaviour
{
    protected Player player;
    protected Rigidbody rigidBody;

    public Vector3 direction { get; private set; }
    public Vector2 lookDelta { get; private set; }
    private Vector2 moveInput; // WASD 입력값을 저장


    [Header("Camera Settings")]
    public Transform cameraTransform; // Inspector에서 할당
    public float mouseSensitivity = 2f;
    public float minPitch = -80f;
    public float maxPitch = 80f;

    [Header("Jump Settings")]
    public float jumpForce = 5f;
    public LayerMask groundLayer;
    public float groundCheckDistance = 0.1f;
    private bool isGrounded = false;

    private float yaw = 0f;
    private float pitch = 0f;

    [Header("Interact Settings")]
    public float interactDistance = 2f;
    public LayerMask interactableLayer;
    private Interactable currentInteractable;
    public void OnInteractInput(InputAction.CallbackContext context)
    {
        if (context.performed && currentInteractable != null)
        {
            currentInteractable.Interact();
        }
    }

    protected const float CONVERT_UNIT_VALUE = 0.01f;
    // 점프 입력 처리 (Input System Button 액션에 연결)

    void Start()
    {
        player = GetComponent<Player>();
        rigidBody = GetComponent<Rigidbody>();

        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;

        // 초기 카메라 각도 저장
        if (cameraTransform != null)
        {
            Vector3 angles = cameraTransform.eulerAngles;
            yaw = angles.y;
            pitch = angles.x;
        }
    }

    public void OnMoveInput(InputAction.CallbackContext context)
    {
        moveInput = context.ReadValue<Vector2>();
    }

    public void OnLookInput(InputAction.CallbackContext context)
    {
        Vector2 delta = context.ReadValue<Vector2>();
        lookDelta = delta;
        ApplyLookDelta(delta);
    }

    public void OnJumpInput(InputAction.CallbackContext context)
    {
        Debug.Log($"Tried Jump | isGrounded: {isGrounded}");
        if (context.performed && isGrounded)
        {
            rigidBody.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
        }
    }

    private void ApplyLookDelta(Vector2 delta)
    {
        if (cameraTransform == null)
            return;

        yaw += delta.x * mouseSensitivity;
        pitch -= delta.y * mouseSensitivity;
        pitch = Mathf.Clamp(pitch, minPitch, maxPitch);

        // 플레이어(본체)는 y축(yaw)만 회전
        transform.rotation = Quaternion.Euler(0f, yaw, 0f);
        // 카메라는 pitch(x축)만 회전, y축은 플레이어와 동기화
        cameraTransform.rotation = Quaternion.Euler(pitch, yaw, 0f);
    }

    // LateUpdate와 HandleCameraRotation은 더 이상 필요 없음 (실시간 처리)

    protected void Move()
    {
        // 카메라 기준으로 이동 방향을 매번 계산
        if (cameraTransform != null)
        {
            Vector3 camForward = cameraTransform.forward;
            Vector3 camRight = cameraTransform.right;
            camForward.y = 0f;
            camRight.y = 0f;
            camForward.Normalize();
            camRight.Normalize();
            direction = (camRight * moveInput.x + camForward * moveInput.y).normalized;
        }
        else
        {
            direction = new Vector3(moveInput.x, 0f, moveInput.y);
        }

        float currentMoveSpeed = player.MoveSpeed * CONVERT_UNIT_VALUE;
        // y축 속도 보존
        Vector3 velocity = direction * currentMoveSpeed;
        velocity.y = rigidBody.linearVelocity.y;
        rigidBody.linearVelocity = velocity;
    }

    void FixedUpdate()
    {
        CheckGrounded();
        Move();
        CheckInteractable();
    }
    // 상호작용 가능한 오브젝트 탐색 및 UI 표시/숨김
    private void CheckInteractable()
    {
        Collider[] hits = Physics.OverlapSphere(transform.position, interactDistance, interactableLayer);
        Interactable nearest = null;
        float minDist = float.MaxValue;
        foreach (var hit in hits)
        {
            Interactable interact = hit.GetComponent<Interactable>();
            if (interact != null)
            {
                float dist = Vector3.Distance(transform.position, hit.transform.position);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearest = interact;
                }
            }
        }
        if (nearest != null)
        {
            if (currentInteractable != nearest)
            {
                if (currentInteractable != null) currentInteractable.HideUI();
                currentInteractable = nearest;
                currentInteractable.ShowUI();
            }
        }
        else
        {
            if (currentInteractable != null)
            {
                currentInteractable.HideUI();
                currentInteractable = null;
            }
        }
    }

    // 바닥 체크 (SphereCast)
    private void CheckGrounded()
    {
        if (rigidBody == null) { isGrounded = false; return; }
        Vector3 origin = transform.position + Vector3.up * -0.4f; // 플레이어 아래쪽으로 origin 이동
        bool hitGround = Physics.SphereCast(origin, 0.2f, Vector3.down, out RaycastHit hit, groundCheckDistance + 0.1f, groundLayer);
        isGrounded = hitGround;
        if (hitGround)
            Debug.Log("Ground hit: " + hit.collider.gameObject.name);
    }
}