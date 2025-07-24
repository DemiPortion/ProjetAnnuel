#include "LockUnlockProvider.h"
#include "LockUnlockCredential.h"
#include <strsafe.h>

LockUnlockProvider::LockUnlockProvider()
    : _cRef(1), _pCredential(nullptr)
{
    InterlockedIncrement(&g_cRefDll);
}

LockUnlockProvider::~LockUnlockProvider()
{
    if (_pCredential)
    {
        _pCredential->Release();
        _pCredential = nullptr;
    }

    InterlockedDecrement(&g_cRefDll);
}

// IUnknown
IFACEMETHODIMP LockUnlockProvider::QueryInterface(REFIID riid, void** ppv)
{
    static const QITAB qit[] = {
        QITABENT(LockUnlockProvider, ICredentialProvider),
        {0}
    };
    return QISearch(this, qit, riid, ppv);
}

IFACEMETHODIMP_(ULONG) LockUnlockProvider::AddRef()
{
    return InterlockedIncrement(&_cRef);
}

IFACEMETHODIMP_(ULONG) LockUnlockProvider::Release()
{
    LONG cRef = InterlockedDecrement(&_cRef);
    if (!cRef)
        delete this;
    return cRef;
}

// ICredentialProvider

IFACEMETHODIMP LockUnlockProvider::SetUsageScenario(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus, DWORD /*dwFlags*/)
{
    if (cpus == CPUS_LOGON || cpus == CPUS_UNLOCK_WORKSTATION)
    {
        if (_pCredential)
        {
            _pCredential->Release();
            _pCredential = nullptr;
        }

        _pCredential = new LockUnlockCredential();
        if (!_pCredential)
            return E_OUTOFMEMORY;

        _pCredential->AddRef();
        return S_OK;
    }
    return E_NOTIMPL;
}

IFACEMETHODIMP LockUnlockProvider::SetSerialization(const CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* /*pcpcs*/)
{
    return E_NOTIMPL;
}

IFACEMETHODIMP LockUnlockProvider::Advise(ICredentialProviderEvents* /*pcpe*/, UINT_PTR /*upAdviseContext*/)
{
    return S_OK;
}

IFACEMETHODIMP LockUnlockProvider::UnAdvise()
{
    return S_OK;
}

IFACEMETHODIMP LockUnlockProvider::GetFieldDescriptorCount(DWORD* pdwCount)
{
    *pdwCount = 3; // Username, Password, Submit button
    return S_OK;
}

IFACEMETHODIMP LockUnlockProvider::GetFieldDescriptorAt(DWORD dwIndex, CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR** ppcpfd)
{
    // Simplification : l’objet `LockUnlockCredential` gère les FDs
    return E_NOTIMPL;
}

IFACEMETHODIMP LockUnlockProvider::GetCredentialCount(DWORD* pdwCount, DWORD* pdwDefault, BOOL* pbAutoLogonWithDefault)
{
    *pdwCount = 1;
    *pdwDefault = 0;
    *pbAutoLogonWithDefault = FALSE;
    return S_OK;
}

IFACEMETHODIMP LockUnlockProvider::GetCredentialAt(DWORD dwIndex, ICredentialProviderCredential** ppcpc)
{
    if (dwIndex == 0 && _pCredential)
    {
        _pCredential->AddRef();
        *ppcpc = _pCredential;
        return S_OK;
    }

    return E_INVALIDARG;
}
