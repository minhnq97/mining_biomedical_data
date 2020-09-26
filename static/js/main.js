function ajax(url, type, json, callback) {
    $.ajax({
        url: url,
        type: type,
        data: JSON.stringify(json),
        contentType: 'application/json; charset=utf-8',
        dataType: 'json',
        success: callback
    });
}

function createWordCloud(selector) {
    const fill = d3.scale.category20();
    const svg = d3.select(selector).append("svg")
        .attr("id", "cloud")
        .attr("width", 500)
        .attr("height", 500)
        .append("g")
        .attr("transform", "translate(250,250)");

    function draw(words) {
        const cloud = svg.selectAll("g text")
            .data(words, function (d) {
                return d.text;
            })
        if (words.length > 0) {
            $("#cloud").css("background", "rgba(255, 255, 255, 0.4)");
        } else {
            $("#cloud").css("background", "rgba(255, 255, 255, 0)");
        }
        cloud.enter()
            .append("text")
            .style("font-family", "Segoe UI Black")
            .style("fill", function (d, i) {
                return fill(i);
            })
            .attr("text-anchor", "middle")
            .attr('font-size', 1)
            .text(function (d) {
                return d.text;
            });
        cloud
            .transition()
            .duration(600)
            .style("font-size", function (d) {
                return d.size + "px";
            })
            .attr("transform", function (d) {
                return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
            })
            .style("fill-opacity", 1);
        cloud.exit()
            .transition()
            .duration(200)
            .style('fill-opacity', 1e-6)
            .attr('font-size', 1)
            .remove();
    }

    return {
        update: function (words) {
            d3.layout.cloud().size([500, 500])
                .words(words)
                .padding(5)
                .rotate(function () {
                    return ~~(Math.random() * 2) * 30;
                })
                .font("Segoe UI Black")
                .fontSize(function (d) {
                    return d.size;
                })
                .on("end", draw)
                .start();
        }
    }
}

const wordCloud = createWordCloud('body');

$(document).on("click", ".category", (e) => {
    $.get("word_cloud", {"category_id": $(e.target).data("id"), "top_n": 200}, function (data) {
        let sum = 0;
        $.each(data, function (key, val) {
            sum += val['doc_count'];
        });
        let words = [];
        $.each(data, function (key, val) {
            words.push({text: val['key'], size: val['doc_count'] / sum * 500});
        });
        wordCloud.update(words);
    }, "json");
});

$("#search").on("keydown", (e) => {
    if (e.keyCode == 13) {
        e.preventDefault();
        $("#ask").click();
    }
});

$("#ask").on("click", (_) => {
    wordCloud.update([]);
    if ($("#form").hasClass("search")) {
        $("#form").toggleClass("search");
        $("#form").toggleClass("result");
    }
    $("#info").text("Bấm vào loại bệnh để xem các từ hay xuất hiện trong triệu chứng bệnh đó dưới góc phải màn hình");
    $("#diagnose").text("Đang chẩn đoán...");
    $("#results").text("Đang tìm kiếm...");
    ajax("classifier", "POST", {"question": $("#search").val()}, function (data) {
        $("#diagnose").html(`Bạn có triệu chứng của bệnh liên quan đến <span data-id="${data["category_id"]}" class="category">${data["category_name"]}</span>.`);
    });
    ajax("similarity", "POST", {"question": $("#search").val(), "top_n": 5}, function (data) {
        let similars = [];
        similars.push(`
            <div class="result-item">
                <div class="similar">Một số bệnh nhân có khả năng cùng triệu chứng:</div>
            </div>`);
        $.each(data, function (key, val) {
            similars.push(`
            <div class="result-item">
                <div class="number">${key + 1}.</div>
                <div class="similar">${val["_source"]["question"]}</div>
                <div class="description">Bệnh chẩn đoán: <span data-id="${val["_source"]["category_id"]}" class="category">${val["_source"]["category"]}</span></div>
            </div>`);
        });
        $("#results").html(similars.join(""));
    });
});
