using UnityEngine;
using UnityEngine.Audio;
using System;
using System.IO;
using Photon.Pun;

[RequireComponent(typeof(AudioListener))]
public class VoiceRecorder : MonoBehaviour
{
    public AudioMixer masterMixer;

    private bool isRecording = false;
    private AudioClip micClip;
    private string micDevice;
    private int sampleRate = 16000; // 원하는 샘플레이트로 변경 가능
    private float originalSfxVolume;
    private float originalMusicVolume;

    private string GetSelectedMicrophone()
    {
        string mic = PlayerPrefs.GetString("SelectedMicrophone", null);
        if (!string.IsNullOrEmpty(mic) && Array.Exists(Microphone.devices, d => d == mic))
            return mic;
        if (Microphone.devices.Length > 0)
            return Microphone.devices[0];
        return null;
    }

    public void StartRecording()
    {
        if (isRecording) return;

        Debug.Log($"[VoiceRecorder] Microphone.Start 방식 녹음 시작. LocalPlayer: {PhotonNetwork.LocalPlayer.NickName}");

        masterMixer.GetFloat("SFXVolume", out originalSfxVolume);
        masterMixer.GetFloat("MusicVolume", out originalMusicVolume);
        masterMixer.SetFloat("SFXVolume", -80f);
        masterMixer.SetFloat("MusicVolume", -80f);

        micDevice = GetSelectedMicrophone();
        if (micDevice != null)
        {
            micClip = Microphone.Start(micDevice, false, 300, sampleRate);
            isRecording = true;
        }
        else
        {
            Debug.LogError("마이크 장치를 찾을 수 없습니다.");
        }
    }

    public void StopRecordingAndSave(string filename)
    {
        if (!isRecording) return;
        isRecording = false;
        int micSamples = Microphone.GetPosition(micDevice); // 실제 녹음된 샘플 수
        Microphone.End(micDevice);

        masterMixer.SetFloat("SFXVolume", originalSfxVolume);
        masterMixer.SetFloat("MusicVolume", originalMusicVolume);

        SaveClipToWav(micClip, filename, micSamples);
    }

    private void SaveClipToWav(AudioClip clip, string filename, int samplesToWrite)
    {
        if (clip == null)
        {
            Debug.LogError("녹음된 AudioClip이 없습니다.");
            return;
        }

        string filepath = Path.Combine(Application.persistentDataPath, filename + ".wav");
        float[] samples = new float[samplesToWrite * clip.channels];
        clip.GetData(samples, 0);

        byte[] wavData = ConvertToWav(samples, clip.channels, clip.frequency);
        File.WriteAllBytes(filepath, wavData);

        Debug.Log("WAV 파일 저장 완료: " + filepath);
    }

    private byte[] ConvertToWav(float[] samples, int channels, int sampleRate)
    {
        int sampleCount = samples.Length;
        int byteCount = sampleCount * 2; // 16bit PCM
        int headerSize = 44;
        byte[] wav = new byte[headerSize + byteCount];

        // WAV 헤더 작성
        // RIFF
        wav[0] = (byte)'R'; wav[1] = (byte)'I'; wav[2] = (byte)'F'; wav[3] = (byte)'F';
        int fileSize = headerSize + byteCount - 8;
        BitConverter.GetBytes(fileSize).CopyTo(wav, 4);
        wav[8] = (byte)'W'; wav[9] = (byte)'A'; wav[10] = (byte)'V'; wav[11] = (byte)'E';
        // fmt chunk
        wav[12] = (byte)'f'; wav[13] = (byte)'m'; wav[14] = (byte)'t'; wav[15] = (byte)' ';
        BitConverter.GetBytes(16).CopyTo(wav, 16); // Subchunk1Size
        BitConverter.GetBytes((short)1).CopyTo(wav, 20); // AudioFormat (PCM)
        BitConverter.GetBytes((short)channels).CopyTo(wav, 22);
        BitConverter.GetBytes(sampleRate).CopyTo(wav, 24);
        BitConverter.GetBytes(sampleRate * channels * 2).CopyTo(wav, 28); // ByteRate
        BitConverter.GetBytes((short)(channels * 2)).CopyTo(wav, 32); // BlockAlign
        BitConverter.GetBytes((short)16).CopyTo(wav, 34); // BitsPerSample
        // data chunk
        wav[36] = (byte)'d'; wav[37] = (byte)'a'; wav[38] = (byte)'t'; wav[39] = (byte)'a';
        BitConverter.GetBytes(byteCount).CopyTo(wav, 40);

        // 샘플 데이터 변환
        for (int i = 0; i < samples.Length; i++)
        {
            short val = (short)(Mathf.Clamp(samples[i], -1f, 1f) * short.MaxValue);
            wav[headerSize + i * 2] = (byte)(val & 0xff);
            wav[headerSize + i * 2 + 1] = (byte)((val >> 8) & 0xff);
        }
        return wav;
    }

    private bool IsTeacher()
    {
        object role;
        if (PhotonNetwork.LocalPlayer.CustomProperties.TryGetValue("Role", out role))
        {
            return role.ToString() == "Teacher";
        }
        return false;
    }

    [PunRPC]
    public void RpcStartRecording()
    {
        Debug.Log($"[VoiceRecorder] RpcStartRecording 호출됨. LocalPlayer: {PhotonNetwork.LocalPlayer.NickName}, Role: {(PhotonNetwork.LocalPlayer.CustomProperties.ContainsKey("Role") ? PhotonNetwork.LocalPlayer.CustomProperties["Role"] : "없음")}");
        if (!IsTeacher())
            StartRecording();
    }

    [PunRPC]
    public void RpcStopRecordingAndSave()
    {
        Debug.Log($"[VoiceRecorder] RpcStopRecordingAndSave 호출됨. LocalPlayer: {PhotonNetwork.LocalPlayer.NickName}, Role: {(PhotonNetwork.LocalPlayer.CustomProperties.ContainsKey("Role") ? PhotonNetwork.LocalPlayer.CustomProperties["Role"] : "없음")}");
        if (!IsTeacher())
            StopRecordingAndSave("VoiceRecord_" + PhotonNetwork.LocalPlayer.NickName);
    }
}