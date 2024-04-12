from js import document
from pyscript import when

@when("change", "#myfile")
async def file_changed(event):
    fileList = event.target.files.to_py()
    for entry in fileList:
        data = await entry.text()
        txt = f'<h2>{entry.name}</h2>\n'
        number = 0
        for line in data.split('\n'):
            number += 1
            txt += f'{number}| {line}<br>'
        print(type(data))
        document.getElementById("content").innerHTML = txt

