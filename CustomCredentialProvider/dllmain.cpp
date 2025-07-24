#include <windows.h>
#include <shlwapi.h>
#include <strsafe.h>
#include "LockUnlockProvider.h"
#include "guid.h"

HINSTANCE g_hInst = NULL;
LONG g_cRefDll = 0;

// Standard DLL entry point
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD dwReason, LPVOID /*lpReserved*/)
{
    if (dwReason == DLL_PROCESS_ATTACH)
    {
        g_hInst = hinstDLL;
        DisableThreadLibraryCalls(hinstDLL);
    }
    return TRUE;
}

// Called by COM to determine if DLL can be unloaded
STDAPI DllCanUnloadNow()
{
    return (g_cRefDll == 0) ? S_OK : S_FALSE;
}

// Factory pattern for instantiating our Provider
class ClassFactory : public IClassFactory
{
public:
    // IUnknown
    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv)
    {
        if (IID_IUnknown == riid || IID_IClassFactory == riid)
        {
            *ppv = static_cast<IClassFactory*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    IFACEMETHODIMP_(ULONG) AddRef() { return InterlockedIncrement(&m_cRef); }
    IFACEMETHODIMP_(ULONG) Release()
    {
        LONG cRef = InterlockedDecrement(&m_cRef);
        if (!cRef)
            delete this;
        return cRef;
    }

    // IClassFactory
    IFACEMETHODIMP CreateInstance(IUnknown* pUnkOuter, REFIID riid, void** ppv)
    {
        if (pUnkOuter)
            return CLASS_E_NOAGGREGATION;

        LockUnlockProvider* pProvider = new LockUnlockProvider();
        if (!pProvider)
            return E_OUTOFMEMORY;

        HRESULT hr = pProvider->QueryInterface(riid, ppv);
        pProvider->Release();
        return hr;
    }

    IFACEMETHODIMP LockServer(BOOL) { return S_OK; }

    ClassFactory() : m_cRef(1) {}
private:
    LONG m_cRef;
};

STDAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv)
{
    if (CLSID_LockUnlockProvider != rclsid)
        return CLASS_E_CLASSNOTAVAILABLE;

    ClassFactory* pFactory = new ClassFactory();
    if (!pFactory)
        return E_OUTOFMEMORY;

    HRESULT hr = pFactory->QueryInterface(riid, ppv);
    pFactory->Release();
    return hr;
}
