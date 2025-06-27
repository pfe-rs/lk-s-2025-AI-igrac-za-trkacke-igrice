# AI Igrač za trkačke igrice

Ovaj repozitorijum je napravljen u akademske svrhe.  
Cilj projekta je razviti i trenirati AI botove koji upravljaju automobilom u trci, i uporediti njihove performanse koristeći različite pristupe.

## Ključne tačke

- Projekat je napravljen u edukativne svrhe
- Cilj je trenirati AI igrače za trkačku igru i uporediti ih
- Korisćeni algoritmi: PPO, genetski algoritam (GA), i učenje uz nadzor (supervizirano)

## Igra

- Auto sa komandama: gas, kočnica, skretanje levo/desno
- Zidovi kao prepreke
- Kontrolne tačke koje se aktiviraju redom


## Setup
```sh
pipenv shell
pip intall -r ./requierments.txt
```

## Algoritmi

### PPO
#### Train
```sh
python -m agent.ppo.trainer
```
#### Run model
```sh
python -m agent.ppo.runner
```
#### Training logs
Taining logs are available through tensor board
```sh
tensorboard --logdir ./logs/ppo_training
```
