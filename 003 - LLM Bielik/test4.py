from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

w_pustyni_i_w_puszczy = f"""
— Wiesz, Nel — mówił Staś Tarkowski do swojej przyjaciółki, małej Angielki — wczoraj przyszli zabtie (policjanci) i aresztowali żonę dozorcy Smaina i jej troje dzieci — tę Fatmę, która już kilka razy przychodziła do biura do twojego ojca i do mego.

A mała, podobna do ślicznego obrazka Nel podniosła swe zielonawe oczy na Stasia i zapytała na wpół ze zdziwieniem, a na wpół ze strachem:

— Wzięli ją do więzienia?

— Nie, ale nie pozwolili jej wyjechać do Sudanu i przyjechał urzędnik, który jej będzie pilnował, by ani krokiem nie wyruszyła z Port-Saidu.

— Dlaczego?

Staś, który kończył rok czternasty i który swą ośmioletnią towarzyszkę kochał bardzo, ale uważał za zupełne dziecko, rzekł z miną wielce zarozumiałą:

— Jak dojdziesz do mego wieku, to będziesz wiedziała wszystko, co się dzieje nie tylko wzdłuż kanału, od Port-Saidu do Suezu, ale i w całym Egipcie. Czy ty nic nie słyszałaś o Mahdim?

— Słyszałam, że jest brzydki i niegrzeczny.

Chłopiec uśmiechnął się z politowaniem.

— Czy jest brzydki — nie wiem. Sudańczycy utrzymują, że jest piękny. Ale powiedzieć, że jest niegrzeczny, o człowieku, który wymordował już tylu ludzi, może tylko dziewczynka ośmioletnia, w sukience, ot! takiej — do kolan!

— Tatuś mi tak powiedział, a tatuś wie najlepiej.

— Powiedział ci tak dlatego, że inaczej byś nie zrozumiała. Do mnie by się tak nie wyraził. Mahdi jest gorszy niż całe stado krokodyli. Rozumiesz? Dobre mi powiedzenie: „niegrzeczny”, tak się mówi do niemowląt.

Lecz ujrzawszy zachmurzoną twarz dziewczynki umilkł, a potem rzekł:

— Nel! wiesz, że nie chciałem ci zrobić przykrości; przyjdzie czas, że i ty będziesz miała czternasty rok. Obiecuję ci to na pewno.

— Aha! — odpowiedziała z zatroskanym wejrzeniem — a jeżeli Mahdi wpadnie przedtem do Port-Saidu i mnie zje?

— Mahdi nie jest ludożercą, więc ludzi nie zjada, tylko ich morduje. Do Port-Saidu też nie wpadnie, a gdyby nawet wpadł i chciał cię zabić, pierwej miałby ze mną do czynienia.

Oświadczenie to oraz świst, z jakim Staś wciągnął nosem powietrze, nie zapowiadający nic dobrego dla Mahdiego, uspokoiły znacznie Nel co do własnej osoby.

— Wiem — odrzekła. — Ty byś mnie nie dał. Ale dlaczego nie puszczają Fatmy z Port-Saidu?

— Bo Fatma jest cioteczną siostrą Mahdiego. Mąż jej, Smain, oświadczył rządowi egipskiemu w Kairze, że pojedzie do Sudanu, gdzie przebywa Mahdi, i wyrobi wolność dla wszystkich Europejczyków, którzy wpadli w jego ręce.

— To Smain jest dobry?

— Czekaj. Twój i mój tatuś, którzy znali doskonale Smaina, nie mieli wcale do niego zaufania i ostrzegali Nubara Paszę, by mu nie ufał. Ale rząd zgodził się wysłać Smaina i Smain bawi od pół roku u Mahdiego. Jeńcy jednak nie tylko nie wrócili, ale przyszła z Chartumu wiadomość, że mahdyści obchodzą się z nimi coraz okrutniej, a że Smain, nabrawszy od rządu pieniędzy, zdradził. Przystał całkiem do Mahdiego i został mianowany emirem. Ludzie powiadają, że w tej okropnej bitwie, w której poległ jenerał Hicks, Smain dowodził artylerią Mahdiego i on to podobno nauczył mahdystów obchodzić się z armatami, czego przedtem, jako dzicy ludzie, wcale nie umieli. Ale Smainowi chodzi teraz o to, by wydostać z Egiptu żonę i dzieci, toteż gdy Fatma, która widocznie z góry wiedziała, co zrobi Smain, chciała cichaczem wyjechać z Port-Saidu, rząd aresztował ją teraz razem z dziećmi.

— A co rządowi przyjdzie z Fatmy i jej dzieci?

— Rząd powie Mahdiemu: „Oddaj nam jeńców, a my oddamy ci Fatmę…”

Na razie rozmowa urwała się, albowiem uwagę Stasia zwróciły ptaki lecące od strony Echtum om Farag ku jezioru Menzaleh. Leciały one dość nisko i w przezroczystym powietrzu widać było wyraźnie kilka pelikanów z zagiętymi na grzbiety szyjami, poruszających z wolna ogromnymi skrzydłami. Staś począł zaraz naśladować ich lot, więc zadarł głowę i biegł przez kilkanaście kroków groblą, machając rozłożonymi rękoma.

— Patrz, lecą i czerwonaki — zawołała nagle Nel.

Staś zatrzymał się w jednej chwili, gdyż istotnie za pelikanami, ale nieco wyżej, widać było zawieszone na błękicie jakby dwa wielkie, różowe i purpurowe kwiaty.

— Czerwonaki! Czerwonaki!

— One wracają pod wieczór do swoich siedzib na wysepkach — rzekł chłopiec. — Ach, gdybym miał strzelbę!

— Po cóż byś miał do nich strzelać?

— Kobiety takich rzeczy nie rozumieją. Ale pójdźmy dalej, może zobaczymy ich więcej.
"""

device = "cuda" # the device to load the model onto

model_name = "speakleash/Bielik-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim."},
    {"role": "user", "content": "Odpowiedz na pytanie, ile nat ma Staś w oparciu o tekst powieści? Podaj wyłącznie liczbę lat w formie pliku JSON { wiek: liczba-lat }. Przykład - jestem Arek i mam 30 lat. Odpowiedź: { wiek: 30 }. Oto tekst powieści: " + w_pustyni_i_w_puszczy}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = input_ids.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

#<s>  Staś ma 14 lat.
#{ wiek: 14 lat }</s>