document.getElementById('visual-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    let feature1 = document.getElementById('feature-1').value
    let feature2 = document.getElementById('feature-2').value

    await fetch(`http://127.0.0.1:5000/getPlot?feature-1=${feature1}&feature-2=${feature2}`, {
        mode: 'cors', // no-cors, *cors, same-origin
    })
        .then(response => response.json())
        .then(data => changeImage(data));
})

function changeImage(data) {
    document.getElementById('image').src = `data:image/png;base64,${data}`
}