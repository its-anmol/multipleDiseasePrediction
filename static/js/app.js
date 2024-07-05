//step 1: get DOM
let nextDom = document.getElementById("next");
let prevDom = document.getElementById("prev");

let carouselDom = document.querySelector(".carousel");
let SliderDom = carouselDom.querySelector(".carousel .list");
let thumbnailBorderDom = document.querySelector(".carousel .thumbnail");
let thumbnailItemsDom = thumbnailBorderDom.querySelectorAll(".item");
let timeDom = document.querySelector(".carousel .time");

thumbnailBorderDom.appendChild(thumbnailItemsDom[0]);
let timeRunning = 3000;
let timeAutoNext = 7000;

nextDom.onclick = function () {
  showSlider("next");
};

prevDom.onclick = function () {
  showSlider("prev");
};
let runTimeOut;
let runNextAuto = setTimeout(() => {
  next.click();
}, timeAutoNext);
function showSlider(type) {
  let SliderItemsDom = SliderDom.querySelectorAll(".carousel .list .item");
  let thumbnailItemsDom = document.querySelectorAll(
    ".carousel .thumbnail .item"
  );

  if (type === "next") {
    SliderDom.appendChild(SliderItemsDom[0]);
    thumbnailBorderDom.appendChild(thumbnailItemsDom[0]);
    carouselDom.classList.add("next");
  } else {
    SliderDom.prepend(SliderItemsDom[SliderItemsDom.length - 1]);
    thumbnailBorderDom.prepend(thumbnailItemsDom[thumbnailItemsDom.length - 1]);
    carouselDom.classList.add("prev");
  }
  clearTimeout(runTimeOut);
  runTimeOut = setTimeout(() => {
    carouselDom.classList.remove("next");
    carouselDom.classList.remove("prev");
  }, timeRunning);

  clearTimeout(runNextAuto);
  runNextAuto = setTimeout(() => {
    next.click();
  }, timeAutoNext);
}

function register(name, email, password) {
  fetch("http://localhost:8000/register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ name, email, password }),
  })
    .then((response) => {
      console.log(response);
      if (response.status == 401) {
        throw new Error("User already exists");
      } else if (response.status == 422) {
        throw new Error("Password must be atleast 5 characters long");
      }
      return response.json();
    })
    .then((data) => {
      alert("Registration successful");
      window.location.href = "./index.html";
    })
    .catch((error) => {
      alert(error);
    });
}

function login(email, password) {
  fetch("http://localhost:8000/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email, password }),
  })
    .then((response) => {
      if (response.status == 401) {
        throw new Error("Incorrect Username or Password");
      } else if (response.status == 422) {
        throw new Error("Password must be atleast 5 characters long");
      }
      return response.json();
    })
    .then((data) => {
      const token = data.token;
      document.cookie = `token=${token};`;
      document.cookie = `email=${email};`;

      alert("Login successful");
      window.location.href = "./index.html";
    })
    .catch((error) => {
      alert(error);
    });
}

function getPredictions() {
  var email = getCookie("email");
  var token = getCookie("token");

  fetch("http://localhost:8000/getprediction", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email: email, token: token }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.length == 0) {
        var info = document.getElementById("info");
        info.textContent = "No Past Predictions found";
      } else {
        var table = document.getElementById("data-table");
        table.style["display"] = "block";
        data.forEach((item) => {
          var row = table.insertRow(-1);
          row.insertCell(0).innerHTML = item.name;
          row.insertCell(1).innerHTML = item.email;
          row.insertCell(2).innerHTML = item.disease;
          var parameters = "";
          for (var key in item.parameters) {
            parameters += key + ": " + item.parameters[key] + "\n";
          }
          parameters = parameters.slice(0, -2);
          row.insertCell(3).innerHTML = parameters;
          row.insertCell(4).innerHTML = item.predictionResult;
        });
      }
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function getCookie(name) {
  var nameEQ = name + "=";
  var ca = document.cookie.split(";");
  for (var i = 0; i < ca.length; i++) {
    var c = ca[i];
    while (c.charAt(0) == " ") c = c.substring(1, c.length);
    if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length, c.length);
  }
  return null;
}
