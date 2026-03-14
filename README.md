# SZTE Multinorm segédkódbázis

A Szegedi Tudományegyetemen folyó, több normával, illetve támadással szemben is robosztus modell(ek) kifejlesztéséhez használt segédkódbázis. Ezzel a kódbázissal több fajta normára/támadásra is robosztus modellek taníthatók és tesztelhetők, amelyek hasonlítási alapot adtak az általunk fejlesztett modell vizsgálatára, teljesítményének lemérésére, és az összehasonlításukra.

Ezt a kódbázist GPL licenszel töltöttem fel, tiszteletben tartva az eredeti tanítási módszereket megalkotó szerzők akaratát, akik munkájukat teljesen open-source alapra helyezték, és elérhetővé tették mindenki számára. Minden egyes módszernél hagyok majd egy github linket az eredeti tanítási módszer repository-jához, ahol meg lehet tekinteni az eredeti munkájukat. Szeretném megjegyezni, hogy ezen kódbázis megalkotásánál a munkájukba nem nyúltam bele, a tanítási módszereik és eredményeik nincsenek alterálva semmilyen módon.

Ha belenyúltam a kódnak egy részébe az annak érdekében történt, hogy tanítási módszerük elérhető legyen más kép-adatbázison is, illetve a Szegedi Tudományegyetemen található belső rendszeren futtatható legyen a kód, akár tanításról, akár kiértékelésről legyen szó.

Szeretném megjegyezni, hogy aki ezt a kódbázist bármiféleképpen módosítja vagy használja, ugyan úgy open-source licensszel kell közzétennie, a licenszfeltételeket tiszteletben tartva.

Ez a segédkódbázis három részre bontható, amelyet a három feltöltött mappa reprezentál. Minden mappába ha belépünk, található egy extra Readme fájl, ami specifikusan elmagyaráz mindent, amiért a mappában található kódok és scriptek a felelősek. A három mappa:

- ***Adatbazis_mappak***: Ide kell a letöltött képadatbázisokat behelyezni. Jelenleg cifar10 és cifar100 képadatbázisokat támogat csak a kód, ezeket kell ide behelyezni. Ezek nem kerülnek a repository-ba feltöltésre, mert nagy méretűek. Akár kézzel töltjük le, akár kóddal, az összes tanító és tesztelő kód amit használ ez a kódbázis ebben a mappában fogja keresni.

- ***Kiertekelo_script***: Ebben a mappában található az összes <ins>kiértékelésre</ins> használt kód és script. Ha a hálók teljesítményét akarjuk lemérni, itt kell keresni.

- ***Vanilla_kodbazisok***: Ebben a mappában található az összes <ins>tanításra</ins> használt kód és script. A tanításokat minden esetben az ebben a mappában lévő scriptekkel kell elindítani.

Ahhoz, hogy a kód futtatható legyen (akár betanításról, akár kiértékelésről van szó) a felhasználónak rendelkeznie kell egy python 3.11-es környezettel, és telepítenie kell a **robustbench** nevű csomagot a python package managerrel (pip). Előfordulhat, hogy egy specifikus tanítási módszerhez szükségesek további csomagok, de ezek mindegyike telepíthető a pip install <csomagnév> paranccsal.

Ha a Szegedi Tudományegyetem clusterén szeretnénk futtatni a kódot, akkor pedig a környezetkezelőben (conda) kell létrehozni egy új környezetet az alábbi parancssorozattal:

```
conda create -n ramp python=3.11
conda activate ramp
pip install robustbench
```

A környezet neve lehet tetszőleges, nem muszáj "ramp"-nek hívni, de akkor az .sh scriptekben ne felejtsük el átírni a környezet nevét (és esetleg helyét).

Ezen felül a tanítás és tesztelés futtatásához is nagyon erősen javasolt a futtatókörnyezetben a megfelelő torch és torchvision verzió feltelepítése ahhoz, hogy kihasználjuk a videókártyák gyorsítását. Az, hogy melyik verzióra lehet szüksége a felhasználónak
nagyban függ attól, hogy milyen videókártyája van, de ha tudjuk a videókártya CUDA verzióját, akkor annak megfelelően lehet választani a következő oldal alapján: https://pytorch.org/get-started/previous-versions/.

(A CUDA verziók lefele kompatibilisek, felfele nem.)
