# Kiertekelo_script mappa

Ebben a mappában találhatóak a kiértékeléshez használt scriptek. A futtatás előfeltétele a környezet beállítása a fő Readme fájlban leírt módon, és az adatbázisok letöltése és behelyezése az Adatbazis_mappak mappába.

A mappában négy kiértékelő script található, **eval_all_HAT**, **eval_all_RAMP**, **eval_all_SPGD**, és **eval_ALL_UNION** néven.

Attól függűen, hogy HAT, RAMP, SPGD vagy UNION modellt szeretnénk tesztelni, az arra létrehozott scriptfájlt kell használni a kiértékeléshez a megfelelő argumentumokkal.

Ha a Szegedi Tudományegyetem clusterén szeretnénk futtatni a kiértékelést, hagytam futtatási mintát az **eval.sh** és az **eval_2.sh** fájlokban, hogy hogyan és milyen argumentumokkal kell elindítani a job-ot a kiértékeléshez.

(Ha UNION modellt szeretnénk cifar10/cifar100-ra tesztelni, szükséges még az union_models mappában a PreactResnet18 architektúrát tartalmazó fájlban a num_class-t 10-re vagy 100-ra átírni, attól függően hogy cifar10 vagy cifar100-as modelleket szeretnénk tesztelni. A többi fajta modellnél ez nem szükséges, azok ezt kezelik.)
