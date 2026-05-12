import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Layout from "../components/Layout";
import { api } from "../lib/api";

export default function StudentDetailPage({ user, onLogout }) {
  const { id } = useParams();
  const navigate = useNavigate();

  const [student, setStudent] = useState(null);
  const [essays, setEssays] = useState([]);
  const [loading, setLoading] = useState(true);

  async function loadStudentDashboard() {
    try {
      setLoading(true);

      const studentsData = await api.get(`/students?userId=${user.id}`);
      const foundStudent = (studentsData.students || []).find(
        (item) => item.id === id
      );

      setStudent(foundStudent || null);

      const reportsData = await api.get(`/reports?userId=${user.id}`);
      const studentEssays = (reportsData.reports || []).filter(
        (report) => report.studentId === id
      );

      setEssays(studentEssays);
    } catch (error) {
      console.error(error);
      alert("Could not load student dashboard.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (user?.id && id) {
      loadStudentDashboard();
    }
  }, [user, id]);

  const scores = essays
    .map((essay) => Number(essay.score || essay.summary?.score || 0))
    .filter((score) => score > 0);

  const averageScore =
    scores.length > 0
      ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1)
      : null;

  return (
    <Layout user={user} onLogout={onLogout} title="Student Dashboard">
      <div className="page-container">
        {loading ? (
          <div className="empty-box">
            <p>Loading student dashboard...</p>
          </div>
        ) : !student ? (
          <div className="empty-box">
            <p>Student not found.</p>
          </div>
        ) : (
          <>
            <button className="text-link" onClick={() => navigate("/students")}>
              ← All students
            </button>

            <div className="student-dashboard-header">
              <div>
                <h1>{student.name}</h1>
                <p>
                  {student.studentNumber || "No student ID"}
                  {student.batch ? ` • ${student.batch}` : ""}
                </p>
                {student.email && <p>{student.email}</p>}
              </div>

              <div className="student-dashboard-stats">
                <span>{essays.length} essays</span>
                <span>
                  {averageScore ? `Avg ${averageScore} / 10` : "No score yet"}
                </span>
              </div>
            </div>

            {essays.length === 0 ? (
              <div className="empty-box">
                <p>No essays evaluated for this student yet.</p>
              </div>
            ) : (
              <div className="essay-history-list">
                {essays.map((essay) => (
                  <div className="essay-history-card" key={essay.id}>
                    <div className="essay-grade-box">
                      {essay.rating === "Excellent"
                        ? "A"
                        : essay.rating === "Good"
                        ? "B"
                        : essay.rating === "Satisfactory"
                        ? "C"
                        : "D"}
                    </div>

                    <div className="essay-history-main">
                      <h3>{essay.title}</h3>
                      <p>
                        {essay.wordCount || 0} words •{" "}
                        {essay.createdAt
                          ? new Date(essay.createdAt).toLocaleString()
                          : "No date"}
                      </p>
                    </div>

                    <div className="essay-score-pill">
                      {essay.score || essay.summary?.score
                        ? `${Number(
                            essay.score || essay.summary?.score
                          ).toFixed(1)} / 10`
                        : essay.rating || "N/A"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </Layout>
  );
}