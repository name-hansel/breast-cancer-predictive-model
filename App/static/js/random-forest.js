document.getElementById('train-model').addEventListener('click', async (e) => {
    e.preventDefault()
    let accuracy = "The accuracy of the model is ";
    await fetch('http://127.0.0.1:5000/trainModel/random-forest', {
        mode: 'cors', // no-cors, *cors, same-origin
    })
        .then(response => response.json())
        .then(data => accuracy += data);

    document.getElementsByClassName('results')[0].style.display = 'flex';
    document.getElementById('accuracy').textContent = accuracy + "%";
    document.getElementById('predict').disabled = false
})

document.getElementById('predict').addEventListener('click', async (e) => {
    e.preventDefault()
    let data = []
    let result;
    let inputs = Array.from(document.getElementsByClassName('input_field'))
    inputs.map(i => data.push(parseFloat(i.value)))

    await fetch('http://127.0.0.1:5000/predict/random-forest', {
        mode: 'cors', // no-cors, *cors, same-origin
        method: 'POST',
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(d => result = d);

    if (result === 'B') {
        document.getElementById('result').textContent = "The tumor is Benign"
    } else if (result === 'M') {
        document.getElementById('result').textContent = "The tumor is Malignant"
    }
})

document.getElementById('reset').addEventListener('click', (e) => {
    document.getElementById('result').textContent = '';
})