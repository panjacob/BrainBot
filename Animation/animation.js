const box = document.getElementById('box')
const car = document.getElementById('car')

// values from -100 to 100
const carSetPosition = (args) => {
    console.log(args)
    const {x, y} = args

    const middleX = window.innerWidth / 2
    const middleY = window.innerHeight / 2
    car.style.left = `${x / 100 * middleX * 0.6 - 50 + middleX}px`
    car.style.top = `${y / 100 * middleY * 0.6 + middleY}px`
}


const createStripe = (x, y) => {
    const stripe = document.createElement('div')
    stripe.style.background = 'white'
    stripe.style.position = 'absolute'
    stripe.style.width = '50px'
    stripe.style.height = '100px'
    stripe.style.top = y + '%'
    stripe.style.left = x + '%'
    box.appendChild(stripe)
    return stripe
}

const roadMove = (speed, forward = true) => {
    const stripeYs = [0, 20, 40, 60, 80, 100]
    const stripes = stripeYs.map(y => createStripe(47, y))

    setInterval(() => {
        stripes.map(stripe => {
            const currentTop = parseInt(stripe.style.top.slice(0, -1))
            // Tutaj tak na szybko niedokładnie policzone
            if (forward) {
                if (currentTop >= 100) stripe.style.top = -20 + '%'
                else stripe.style.top = currentTop + 1 + '%'
            } else {
                if (currentTop < -20) stripe.style.top = 100 + '%'
                else stripe.style.top = currentTop - 1 + '%'
            }
        })
    }, speed);

}


const moveCar = (left) => {
    if (left) carSetPosition({'x': -70, 'y': 50})
    else carSetPosition({'x': 70, 'y': 50})
}

const readAPI = () => {
    const left = true
    const forward = true
    moveCar(left)
    roadMove(25, forward)
}

readAPI()