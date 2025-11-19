using System;
using System.Collections.Generic;
using ExitGames.Client.Photon;
using Photon.Pun;
using Photon.Realtime;
using UnityEngine;

/// <summary>
/// 런타임 플레이어 오브젝트에 Firestore 로드아웃을 적용합니다.
/// </summary>
[DisallowMultipleComponent]
public class PlayerLoadoutApplier : MonoBehaviourPunCallbacks
{
    [SerializeField] private CustomizeManager customizeManager;

    private PhotonView cachedView;

    private void Awake()
    {
        if (customizeManager == null)
        {
            customizeManager = GetComponentInChildren<CustomizeManager>();
        }

        cachedView = GetComponent<PhotonView>();
        if (cachedView == null)
        {
            cachedView = GetComponentInParent<PhotonView>();
        }
    }

    private void Start()
    {
        PhotonView view = GetPhotonView();
        if (view == null)
        {
            Debug.LogWarning("PlayerLoadoutApplier: PhotonView를 찾지 못했습니다.");
            return;
        }

        if (!view.IsMine)
        {
            ApplyLoadoutFromProperties();
        }
    }

    public void ApplyLocalLoadout(Dictionary<string, string> loadout)
    {
        PhotonView view = GetPhotonView();
        if (customizeManager == null || view == null)
        {
            Debug.LogWarning("PlayerLoadoutApplier: Cannot apply local loadout (CustomizeManager or PhotonView missing).");
            return;
        }

        if (loadout == null || loadout.Count == 0)
        {
            Debug.Log("PlayerLoadoutApplier: Local loadout snapshot empty, trying Photon properties fallback.");
            ApplyLoadoutFromProperties();
            return;
        }

        Dictionary<string, object> payload = BuildPayload(loadout);
        Debug.Log($"PlayerLoadoutApplier: Applying local loadout for player {view.Owner?.ActorNumber ?? -1}. itemCount={payload.Count}");
        _ = customizeManager.ApplyLoadoutAsync(payload);
    }

    private void ApplyLoadoutFromProperties()
    {
        PhotonView view = GetPhotonView();
        if (customizeManager == null || view == null)
        {
            return;
        }

        Player owner = view.Owner;
        if (owner == null || owner.CustomProperties == null)
        {
            return;
        }

        if (!owner.CustomProperties.TryGetValue("loadout", out object raw) || raw is not string json || string.IsNullOrWhiteSpace(json))
        {
            return;
        }

        try
        {
            Dictionary<string, string> loadout = LoadoutSerializer.Deserialize(json);
            if (loadout == null || loadout.Count == 0)
            {
                Debug.Log("PlayerLoadoutApplier: Deserialized loadout was null or empty.");
                return;
            }

            Dictionary<string, object> payload = BuildPayload(loadout);
            Debug.Log($"PlayerLoadoutApplier: Applying Photon loadout for player {owner.ActorNumber}. itemCount={payload.Count}");
            _ = customizeManager.ApplyLoadoutAsync(payload);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"PlayerLoadoutApplier: 프로퍼티 로드아웃 파싱 실패 - {ex.Message}");
        }
    }

    public void OnLoadoutPropertyChanged()
    {
        ApplyLoadoutFromProperties();
    }

    public override void OnPlayerPropertiesUpdate(Player targetPlayer, Hashtable changedProps)
    {
        PhotonView view = GetPhotonView();
        if (view == null)
        {
            return;
        }

        if (targetPlayer != view.Owner || changedProps == null || !changedProps.ContainsKey("loadout"))
        {
            return;
        }

        ApplyLoadoutFromProperties();
    }

    private Dictionary<string, object> BuildPayload(Dictionary<string, string> loadout)
    {
        Dictionary<string, object> payload = new Dictionary<string, object>(loadout.Count, StringComparer.OrdinalIgnoreCase);
        foreach (KeyValuePair<string, string> entry in loadout)
        {
            if (!string.IsNullOrWhiteSpace(entry.Key) && !string.IsNullOrWhiteSpace(entry.Value))
            {
                payload[entry.Key] = entry.Value;
            }
        }

        return payload;
    }

    private PhotonView GetPhotonView()
    {
        if (cachedView == null)
        {
            cachedView = GetComponent<PhotonView>();
            if (cachedView == null)
            {
                cachedView = GetComponentInParent<PhotonView>();
            }
        }

        return cachedView;
    }
}
