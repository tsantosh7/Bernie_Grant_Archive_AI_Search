function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').style.display = 'flex';
}

function closeLightbox() {
  document.getElementById('lightbox').style.display = 'none';
}

// Show or hide the scroll-to-top button based on scroll position
window.addEventListener('scroll', function () {
  const upButton = document.getElementById("upButton");
  if (window.scrollY > 100) {
    upButton.style.display = "block";
  } else {
    upButton.style.display = "none";
  }
});

// Function to scroll to the top smoothly
function scrollToTop() {
  document.getElementById("top").scrollIntoView({ behavior: 'smooth' });
}

// Optional smooth scroll for citation links that jump to sources on the same page
document.addEventListener("DOMContentLoaded", function () {
  const citationLinks = document.querySelectorAll(".citation-link");
  citationLinks.forEach(link => {
    link.addEventListener("click", function (event) {
      event.preventDefault();
      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
});
