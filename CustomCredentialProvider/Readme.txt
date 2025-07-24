CustomCredentialProvider – Credential Provider Personnalisé Windows 10/11
==========================================================================

Ce projet est un exemple de Credential Provider personnalisé écrit en C++.

Fonctionnalités :
-----------------
- Interface simplifiée (username, password, bouton).
- Verrouillage automatique via LockWorkStation().
- Déverrouillage conditionné par la validation d’un script Python externe.
- Compatible Windows 10 et 11 (x64).

Compilation :
-------------
1. Ouvre CustomCredentialProvider.sln dans Visual Studio 2019 ou 2022.
2. Compile en Debug ou Release x64.
3. Le fichier généré est : CustomCredentialProvider.dll

Enregistrement :
----------------
1. Registre : double-cliquer sur `registry.reg` ou lancer via `reg import registry.reg`
2. Enregistrement de la DLL : `regsvr32 CustomCredentialProvider.dll` (en administrateur)

Test :
------
- Redémarre Windows ou ferme ta session.
- À l’écran de connexion, la tuile personnalisée doit apparaître.

Signature :
-----------
Si tu veux signer ta DLL avec signtool :

    signtool sign /fd SHA256 /a /tr http://timestamp.digicert.com /td SHA256 CustomCredentialProvider.dll