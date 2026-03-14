# Vanilla_kodbazisok mappa

Ebben a mappában a tanításhoz szükséges scriptek találhatóak. Attól függően, hogy milyen modellt szeretnénk tanítani, értelemszerűen a modell mappájában lévő scripteket kell használni. A következőekben részletezem mind a négy módszernél, hogyan kell egy tanítást elindítani.

## **HAT** (Helper-based Adverserial Training):

Az eredeti kód github repository-ja: https://github.com/imrahulr/hat

Ez a fajta tanítóalgoritmus két részből áll. Először, a tanitas.sh script alapján előállítunk egy "helper" modellt, utána pedig a tényleges tanítás a tanitas_part2.sh fájl alapján történik. Ez használni fogja a helper modellt, hogy előállítson egy olyan új modellt, amelyik robosztus többféle perturbációval szemben is. Az állítható argumentumokat az eredeti github repository README-jében lehet elolvasni.


## **RAMP** (Robustness Against Multiple Perturbations)

Az eredeti kód github repository-ja: https://github.com/uiuc-focal-lab/RAMP

Ez a fajta tanítóalgoritmus egy olyan modellt állít elő, amely három fajta perturbáció ellen is robosztus (Linf,L2,L1). A tanitas.sh fájlban található futtatási minta, a használható argumentumokat pedig az eredeti github repository README-jében el lehet érni.


## **SparsePGD**

Az eredeti kód github repository-ja: https://github.com/CityU-MLO/sPGD

Ez a fajta tanítóalgoritmus egy nem-konvenciós támadás ellen (L0 norma) állít elő robosztus modellt. Több normával szemben nem robosztus, azonban többfajta típusú (untargeted/targeted, white/black-box) L0-normájú támadásokkal szemben is robosztus.


## **Union**

Az eredeti kód github repository_ja: https://github.com/locuslab/robust_union/tree/master

Ez egy régebbi módszer a többihez képest, de ez a fajta módszer is olyan módellt állít elő, amely többféle perturbációval szemben is robosztus. A megfelelő kapcsoló használatával (-model 1-7) többféle típusú modell is előállítható. Ennél is hagytam futtatási mintát a tanitas.sh és tanitas_cifar100.sh fájlokban.

Itt szeretném megjegyezni, hogy a 7-es fajta modell előállítása (RANDOM modell) nincsen benne az eredeti kódban, az általam lett implementálva. A működésének lényege azonban az általuk definiált háromfajta támadás közötti véletlenszerű választás tanító-batchenként (hogy alteráljon a támadások között a tanulás során), ez újfajta módszert nem vezet be a kódjukba, csak a már létező, meglévő módszereket használja másféleképpen.
