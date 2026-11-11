
from pathlib import Path
from collections import defaultdict

INPUT_FOLDER = Path("output/sentences")
OUTPUT_FOLDER = Path("output/location_groups")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

locations = {
    "Garhmukteshwar": [
        "गढ़ मुक्तेश्वर", "गढ़मुक्तेश्वर",
        "शिव बल्लभपुर", "मुक्तेश्वर"
    ],
    "Haridwar": [
        "हरिद्वार", "गंगा द्वार", "लक्सर"
    ],
    "Ujjain": [
        "उज्जैन", "महाकाल", "शिप्रा", "क्षिप्रा"
    ],
    "Ayodhya": [
        "अयोध्या", "सरयू"
    ],
    "Mathura": [
        "मथुरा", "यमुना तट"
    ],
    "Kanchipuram": [
        "कांचीपुरम", "कांजीवरम", "कांची"
    ],
    "Varanasi": [
        "बनारस", "वाराणसी", "काशी", "मुगलसराय"
    ],
    "Dwarka": [
        "द्वारका"
    ]
}

for file_path in sorted(INPUT_FOLDER.glob("*.txt")):
    sentences = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    groups = defaultdict(list)
    uncertain = []

    for sentence in sentences:
        matched = False

        for topic, keywords in locations.items():
            if any(keyword in sentence for keyword in keywords):
                groups[topic].append(sentence)
                matched = True

        if not matched:
            uncertain.append(sentence)

    output = ""

    for topic, group in groups.items():
        output += f"\nTOPIC: {topic}\n"
        output += "\n".join(group) + "\n"

    output += "\nTOPIC: UNCERTAIN\n"
    output += "\n".join(uncertain) + "\n"

    (OUTPUT_FOLDER / file_path.name).write_text(
        output, encoding="utf-8"
    )

    print(file_path.name, "Topics:", len(groups),
          "Uncertain:", len(uncertain))

print("Location grouping completed!")
