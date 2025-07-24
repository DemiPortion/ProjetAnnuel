#include "LockUnlockCredential.h"
#include <strsafe.h>
#include <windows.h>
#include <wincred.h>
#include <credentialprovider.h>
#include <shlwapi.h>
#include <iostream>
#include <sstream>

#pragma comment(lib, "Shlwapi.lib")

LockUnlockCredential::LockUnlockCredential()
    : _cRef(1), _pszUsername(nullptr), _pszPassword(nullptr)
{
    InterlockedIncrement(&g_cRefDll);
}

LockUnlockCredential::~LockUnlockCredential()
{
    CoTaskMemFree(_pszUsername);
    CoTaskMemFree(_pszPassword);

    InterlockedDecrement(&g_cRefDll);
}

// IUnknown
IFACEMETHODIMP LockUnlockCredential::QueryInterface(REFIID riid, void** ppv)
{
    static const QITAB qit[] = {
        QITABENT(LockUnlockCredential, ICredentialProviderCredential),
        {0}
    };
    return QISearch(this, qit, riid, ppv);
}

IFACEMETHODIMP_(ULONG) LockUnlockCredential::AddRef()
{
    return InterlockedIncrement(&_cRef);
}

IFACEMETHODIMP_(ULONG) LockUnlockCredential::Release()
{
    LONG cRef = InterlockedDecrement(&_cRef);
    if (!cRef)
        delete this;
    return cRef;
}

// ICredentialProviderCredential

IFACEMETHODIMP LockUnlockCredential::Advise(ICredentialProviderCredentialEvents* /*pcpe*/) { return S_OK; }
IFACEMETHODIMP LockUnlockCredential::UnAdvise() { return S_OK; }

IFACEMETHODIMP LockUnlockCredential::SetSelected(BOOL* pbAutoLogon)
{
    *pbAutoLogon = FALSE;
    return S_OK;
}

IFACEMETHODIMP LockUnlockCredential::SetDeselected() { return S_OK; }

IFACEMETHODIMP LockUnlockCredential::GetFieldState(DWORD dwFieldID, CREDENTIAL_PROVIDER_FIELD_STATE* pcpfs, CREDENTIAL_PROVIDER_FIELD_INTERACTIVE_STATE* pcpfis)
{
    switch (dwFieldID)
    {
        case 0: // Username
        case 1: // Password
            *pcpfs = CPFS_DISPLAY_IN_SELECTED_TILE;
            *pcpfis = CPFIS_FOCUSED;
            break;
        case 2: // Submit button
            *pcpfs = CPFS_DISPLAY_IN_SELECTED_TILE;
            *pcpfis = CPFIS_NONE;
            break;
        default:
            return E_INVALIDARG;
    }
    return S_OK;
}

IFACEMETHODIMP LockUnlockCredential::GetStringValue(DWORD dwFieldID, PWSTR* ppsz)
{
    if (dwFieldID == 0)
        return SHStrDupW(_pszUsername ? _pszUsername : L"", ppsz);
    return SHStrDupW(L"", ppsz);
}

IFACEMETHODIMP LockUnlockCredential::GetBitmapValue(DWORD, HBITMAP*) { return E_NOTIMPL; }
IFACEMETHODIMP LockUnlockCredential::GetCheckboxValue(DWORD, BOOL*, PWSTR*) { return E_NOTIMPL; }
IFACEMETHODIMP LockUnlockCredential::GetComboBoxValueCount(DWORD, DWORD*, DWORD*) { return E_NOTIMPL; }
IFACEMETHODIMP LockUnlockCredential::GetComboBoxValueAt(DWORD, DWORD, PWSTR*) { return E_NOTIMPL; }
IFACEMETHODIMP LockUnlockCredential::GetSubmitButtonValue(DWORD, DWORD* pdwAdjacentTo)
{
    *pdwAdjacentTo = 1; // Adjacent to password
    return S_OK;
}

IFACEMETHODIMP LockUnlockCredential::SetStringValue(DWORD dwFieldID, LPCWSTR psz)
{
    switch (dwFieldID)
    {
        case 0:
            CoTaskMemFree(_pszUsername);
            _pszUsername = _wcsdup(psz);
            break;
        case 1:
            CoTaskMemFree(_pszPassword);
            _pszPassword = _wcsdup(psz);
            break;
        default:
            return E_INVALIDARG;
    }
    return S_OK;
}

static bool RunPythonValidator(const std::wstring& username, const std::wstring& password)
{
    std::wstringstream cmd;
    cmd << L"python \"C:\\Path\\To\\validate.py\" \"" << username << L"\" \"" << password << L"\"";

    FILE* pipe = _wpopen(cmd.str().c_str(), L"rt");
    if (!pipe) return false;

    wchar_t buffer[128];
    std::wstring output;
    while (fgetws(buffer, 128, pipe)) {
        output += buffer;
    }
    _pclose(pipe);

    return output.find(L"OK") != std::wstring::npos;
}

IFACEMETHODIMP LockUnlockCredential::GetSerialization(
    CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* pcpcs,
    PWSTR* ppszOptionalStatusText,
    CREDENTIAL_PROVIDER_STATUS_ICON* pcpsiOptionalStatusIcon)
{
    if (!_pszUsername || !_pszPassword)
    {
        *ppszOptionalStatusText = _wcsdup(L"Veuillez remplir tous les champs.");
        *pcpsiOptionalStatusIcon = CPSI_ERROR;
        return E_FAIL;
    }

    if (!RunPythonValidator(_pszUsername, _pszPassword))
    {
        *ppszOptionalStatusText = _wcsdup(L"Authentification externe échouée.");
        *pcpsiOptionalStatusIcon = CPSI_ERROR;
        return E_FAIL;
    }

    UNICODE_STRING usUser, usPass;
    RtlInitUnicodeString(&usUser, _pszUsername);
    RtlInitUnicodeString(&usPass, _pszPassword);

    KERB_INTERACTIVE_UNLOCK_LOGON kiul = {};
    kiul.MessageType = KerbInteractiveLogon;
    kiul.Logon.DomainName.Length = 0;
    kiul.Logon.DomainName.Buffer = nullptr;
    kiul.Logon.UserName = usUser;
    kiul.Logon.Password = usPass;

    DWORD cb = sizeof(KERB_INTERACTIVE_UNLOCK_LOGON);
    BYTE* buffer = (BYTE*)CoTaskMemAlloc(cb);
    memcpy(buffer, &kiul, cb);

    pcpcs->ulAuthenticationPackage = 0; // Will be filled by LSASS
    pcpcs->clsidCredentialProvider = CLSID_LockUnlockProvider;
    pcpcs->cbSerialization = cb;
    pcpcs->rgbSerialization = buffer;

    *ppszOptionalStatusText = nullptr;
    *pcpsiOptionalStatusIcon = CPSI_SUCCESS;

    return S_OK;
}

IFACEMETHODIMP LockUnlockCredential::ReportResult(NTSTATUS, NTSTATUS, PWSTR* ppszOptionalStatusText, CREDENTIAL_PROVIDER_STATUS_ICON* pcpsiOptionalStatusIcon)
{
    *ppszOptionalStatusText = nullptr;
    *pcpsiOptionalStatusIcon = CPSI_NONE;
    return S_OK;
}
